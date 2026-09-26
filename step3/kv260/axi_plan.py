"""AXI4 INCR descriptor contract for an eventual KV260 MMU master.

An 8 KiB software page is not an AXI burst: split at 4 KiB boundaries and at
256 beats. This module is an executable specification, NOT a DMA driver.
"""
from dataclasses import asdict, dataclass
import argparse
import json


@dataclass(frozen=True)
class Burst:
    address: int
    beats: int
    beat_bytes: int

    @property
    def arlen(self):
        return self.beats - 1

    @property
    def arsize(self):
        return self.beat_bytes.bit_length() - 1


def split_read(address, nbytes, beat_bytes=16, max_beats=256, address_bits=49):
    """Aligned full-beat requests only. The caller reassembles responses in order."""
    if any(type(x) is not int for x in (address, nbytes, beat_bytes, max_beats, address_bits)):
        raise ValueError("all descriptor parameters must be integers")
    if beat_bytes not in (4, 8, 16) or not 1 <= max_beats <= 256 or not 1 <= address_bits <= 64:
        raise ValueError("invalid AXI parameters")
    if address < 0 or nbytes <= 0 or address % beat_bytes or nbytes % beat_bytes:
        raise ValueError("require nonnegative aligned address and positive whole-beat length")
    if address + nbytes > 1 << address_bits:
        raise ValueError("DMA address range overflows the configured address width")
    out = []
    while nbytes:
        size = min(nbytes, 4096 - address % 4096, max_beats * beat_bytes)
        out.append(Burst(address, size // beat_bytes, beat_bytes))
        address += size
        nbytes -= size
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=lambda s: int(s, 0), default=0,
                        help="DMA device address; default 0 is only a relative example")
    parser.add_argument("--bytes", type=int, default=8192)
    parser.add_argument("--max-beats", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps([dict(**asdict(b), arlen=b.arlen, arsize=b.arsize)
                      for b in split_read(args.base, args.bytes, max_beats=args.max_beats)], indent=2))


if __name__ == "__main__":
    main()
