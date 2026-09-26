"""Python mirror of uapi/kv260_accel.h for host tests and future tooling.

Shares ioctl struct layouts via ctypes. Not a substitute for the C UAPI header.
"""
from __future__ import annotations

import ctypes
from ctypes import c_uint32, c_uint64, c_int32

IOCTL_MAGIC = 0x4B
ABI_VERSION = 1
ISA_ADDR_SHIFT = 6
SOFT_PAGE_BYTES = 8192
AXI_BOUNDARY_BYTES = 4096
AXI_MAX_BEATS = 256

# Register placeholders: TBD until bitstream + DTS
REG_CTRL = -1
REG_STATUS = -1
REG_IMAGE_BASE_LO = -1
REG_IMAGE_BASE_HI = -1

CAP_HAS_DT = 1 << 0
CAP_HAS_REG = 1 << 1
CAP_HAS_IRQ = 1 << 2
CAP_HAS_DMA = 1 << 3
CAP_IMAGE_BASE_SET = 1 << 4
CAP_BITSTREAM_HINT = 1 << 5
CAP_BOARD_PROBE_OK = 1 << 6


class Version(ctypes.Structure):
    _fields_ = [
        ("major", c_uint32),
        ("minor", c_uint32),
        ("patch", c_uint32),
        ("abi", c_uint32),
    ]


class Caps(ctypes.Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("reserved0", c_uint32),
        ("max_buffers", c_uint32),
        ("isa_addr_shift", c_uint32),
        ("max_buffer_bytes", c_uint64),
        ("page_bytes", c_uint64),
        ("features", c_uint64),
        ("image_base_dma", c_uint64),
    ]


class AllocBuffer(ctypes.Structure):
    _fields_ = [
        ("size", c_uint64),
        ("flags", c_uint64),
        ("handle", c_uint64),
        ("dma_addr", c_uint64),
        ("cpu_mmap_offset", c_uint64),
    ]


class FreeBuffer(ctypes.Structure):
    _fields_ = [
        ("handle", c_uint64),
        ("reserved", c_uint64),
    ]


class SetImageBase(ctypes.Structure):
    _fields_ = [
        ("handle", c_uint64),
        ("dma_addr", c_uint64),
        ("flags", c_uint64),
        ("reserved", c_uint64),
    ]


class SubmitDesc(ctypes.Structure):
    _fields_ = [
        ("handle", c_uint64),
        ("nbytes", c_uint64),
        ("flags", c_uint64),
        ("status", c_int32),
        ("reserved", c_uint32),
    ]


class ProbeInfo(ctypes.Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("kernel_has_cma", c_uint32),
        ("note_flags", c_uint32),
        ("reserved0", c_uint32),
        ("reserved1", c_uint64),
        ("reserved2", c_uint64),
    ]


def isa_to_phys(image_base_dma: int, isa_addr: int) -> int:
    """physical = IMAGE_BASE + (instruction.addr << 6). IMAGE_BASE is DMA device addr."""
    if image_base_dma < 0 or isa_addr < 0:
        raise ValueError("addresses must be non-negative")
    return int(image_base_dma) + (int(isa_addr) << ISA_ADDR_SHIFT)


def _ioc(dir_, nr, size):
    # Linux ioctl encoding (generic): dir << 30 | type << 8 | nr | size << 16
    # On Linux: _IOC(dir,type,nr,size) = dir<<30 | type<<8 | nr | size<<16
    return (dir_ << 30) | (IOCTL_MAGIC << 8) | nr | (size << 16)


_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2
_IOC_READWRITE = _IOC_READ | _IOC_WRITE

IOCTL_GET_VERSION = _ioc(_IOC_READ, 0x01, ctypes.sizeof(Version))
IOCTL_QUERY_CAPS = _ioc(_IOC_READ, 0x02, ctypes.sizeof(Caps))
IOCTL_ALLOC_BUFFER = _ioc(_IOC_READWRITE, 0x03, ctypes.sizeof(AllocBuffer))
IOCTL_FREE_BUFFER = _ioc(_IOC_WRITE, 0x04, ctypes.sizeof(FreeBuffer))
IOCTL_SET_IMAGE_BASE = _ioc(_IOC_WRITE, 0x05, ctypes.sizeof(SetImageBase))
IOCTL_SUBMIT_DESC = _ioc(_IOC_READWRITE, 0x06, ctypes.sizeof(SubmitDesc))
IOCTL_RUN_PROBE = _ioc(_IOC_READ, 0x07, ctypes.sizeof(ProbeInfo))
