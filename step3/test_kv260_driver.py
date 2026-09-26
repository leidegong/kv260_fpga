"""Register-map and host-driver regression; uses the fake device, never hardware."""
import unittest

import numpy as np

from kv260 import regmap
from kv260.bw_driver import BwTest, FakeBwDevice


class RegmapTests(unittest.TestCase):
    def test_generated_package_is_current(self):
        self.assertEqual(regmap.PKG.read_text(encoding="utf-8"), regmap.sv_package(),
                         "run: python kv260/regmap.py --write")

    def test_offsets_unique_and_aligned(self):
        for regs in (regmap.BW_REGS, regmap.ACC_REGS):
            offsets = [r.offset for r in regs]
            self.assertEqual(len(offsets), len(set(offsets)))
            self.assertTrue(all(o % 4 == 0 and o < 4096 for o in offsets))


class BwDriverTests(unittest.TestCase):
    def setUp(self):
        self.buf = np.zeros(1 << 20, np.uint8)
        self.phys = 0x7_0000_0000          # >4 GiB device address exercises the HI register
        self.dev = FakeBwDevice(self.buf, self.phys)
        self.test = BwTest(self.dev, self.buf, self.phys, 200e6)

    def test_write_then_read_checksum(self):
        w = self.test.write_pattern(4096, 65536, 0xFFFFFF00)       # wraps the 32-bit pattern
        r = self.test.read(4096, 65536, repeat=3)
        self.assertEqual(w["bytes"], 65536)
        self.assertEqual(r["bytes"] * r["repeat"], 3 * 65536)
        self.assertEqual(self.dev.regs["RD_ADDR_HI"], 7)

    def test_detects_corruption(self):
        self.test.write_pattern(0, 4096, 1)
        orig = self.dev.write

        def corrupt(off, value):
            orig(off, value)
            if off == regmap.BW_REGS[2].offset and value & 1:
                self.dev.regs["RD_SUM"] ^= 1
        self.dev.write = corrupt
        with self.assertRaises(RuntimeError):
            self.test.read(0, 4096)

    def test_rejects_bad_ranges_and_id(self):
        with self.assertRaises(ValueError):
            self.test.read(16, 4096)                               # not 64 B aligned for 4 ports
        with self.assertRaises(ValueError):
            self.test.write_pattern(0, len(self.buf) + 16, 0)
        self.dev.regs["ID"] = 0
        with self.assertRaises(RuntimeError):
            BwTest(self.dev, self.buf, self.phys, 200e6)


if __name__ == "__main__":
    unittest.main()
