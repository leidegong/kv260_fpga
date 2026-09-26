import unittest

from kv260.axi_plan import split_read, split_write


class AxiPlanTests(unittest.TestCase):
    def test_pages_offsets_and_burst_limits(self):
        for beat in (4, 8, 16):
            for base in (0, beat, 4096 - beat, 8192, 0x100000000 + 16):
                for length in (beat, 4096, 8192, 3 * 8192):
                    for limit in (16, 64, 256):
                        bursts = split_read(base, length, beat, limit)
                        cursor = base
                        for b in bursts:
                            size = b.beats * b.beat_bytes
                            self.assertEqual(b.address, cursor)
                            self.assertEqual(b.address // 4096, (b.address + size - 1) // 4096)
                            self.assertTrue(1 <= b.beats <= limit)
                            self.assertEqual(1 << b.arsize, beat)
                            self.assertEqual(b.arlen + 1, b.beats)
                            cursor += size
                        self.assertEqual(cursor, base + length)

    def test_split_write_mirrors_read(self):
        for beat in (4, 8, 16):
            for base in (0, beat, 4096 - beat, 8192):
                for length in (beat, 4096, 8192):
                    for limit in (16, 256):
                        self.assertEqual(
                            split_write(base, length, beat, limit),
                            split_read(base, length, beat, limit),
                        )

    def test_reject_bad_alignment_and_overflow(self):
        for args in ((1, 8192), (0, 0), (0, -16), (0, 17), ((1 << 49) - 16, 32)):
            with self.assertRaises(ValueError):
                split_read(*args)


if __name__ == "__main__":
    unittest.main()
