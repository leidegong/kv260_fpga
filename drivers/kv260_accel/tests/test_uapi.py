#!/usr/bin/env python3
"""Host-only tests: UAPI struct sizes/layout and IMAGE_BASE address helper.

No KV260, no kernel module, no MMIO. Run from drivers/kv260_accel:
  python3 tests/test_uapi.py
  make test
"""
from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "userspace"))

import kv260_accel_uapi as uapi  # noqa: E402


class TestStructSizes(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(ctypes.sizeof(uapi.Version), 16)
        self.assertEqual(ctypes.sizeof(uapi.Caps), 48)
        self.assertEqual(ctypes.sizeof(uapi.AllocBuffer), 40)
        self.assertEqual(ctypes.sizeof(uapi.FreeBuffer), 16)
        self.assertEqual(ctypes.sizeof(uapi.SetImageBase), 32)
        self.assertEqual(ctypes.sizeof(uapi.SubmitDesc), 32)
        self.assertEqual(ctypes.sizeof(uapi.ProbeInfo), 32)

    def test_constants(self):
        self.assertEqual(uapi.ISA_ADDR_SHIFT, 6)
        self.assertEqual(uapi.SOFT_PAGE_BYTES, 8192)
        self.assertEqual(uapi.AXI_BOUNDARY_BYTES, 4096)
        self.assertEqual(uapi.AXI_MAX_BEATS, 256)
        self.assertEqual(uapi.REG_CTRL, -1)
        self.assertEqual(uapi.REG_IMAGE_BASE_LO, -1)
        self.assertEqual(uapi.ABI_VERSION, 1)

    def test_ioctl_numbers_nonzero(self):
        for name in (
            "IOCTL_GET_VERSION",
            "IOCTL_QUERY_CAPS",
            "IOCTL_ALLOC_BUFFER",
            "IOCTL_FREE_BUFFER",
            "IOCTL_SET_IMAGE_BASE",
            "IOCTL_SUBMIT_DESC",
            "IOCTL_RUN_PROBE",
        ):
            self.assertTrue(getattr(uapi, name), name)


class TestIsaToPhys(unittest.TestCase):
    def test_vectors(self):
        cases = [
            (0, 0, 0),
            (0x1000, 1, 0x1000 + 64),
            (0x80000000, 0x10, 0x80000000 + (0x10 << 6)),
            (0x100000000, 2, 0x100000000 + 128),
            (0xA0000000, 0, 0xA0000000),  # base may look "physical"; still only DMA example
        ]
        for base, addr, expect in cases:
            self.assertEqual(uapi.isa_to_phys(base, addr), expect)

    def test_shift_contract(self):
        # Spec: physical = IMAGE_BASE + (instruction.addr << 6)
        base = 0x40000000
        for addr in (0, 1, 63, 64, 1000, 0x10000):
            self.assertEqual(uapi.isa_to_phys(base, addr), base + (addr << 6))

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            uapi.isa_to_phys(-1, 0)
        with self.assertRaises(ValueError):
            uapi.isa_to_phys(0, -1)


class TestAxiPlanReference(unittest.TestCase):
    """Driver docs must point at axi_plan; do not fork split logic here."""

    def test_import_axi_plan(self):
        step3 = ROOT.parents[1] / "step3"
        sys.path.insert(0, str(step3))
        from kv260.axi_plan import split_read  # noqa: WPS433

        bursts = split_read(0, 8192, beat_bytes=16, max_beats=256)
        self.assertEqual(len(bursts), 2)
        self.assertEqual(bursts[0].beats, 256)
        self.assertEqual(bursts[1].beats, 256)
        self.assertEqual(sum(b.beats * b.beat_bytes for b in bursts), 8192)


if __name__ == "__main__":
    unittest.main()
