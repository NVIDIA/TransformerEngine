# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Host-only checks for fractional byte sizes and reported locality boundaries."""

import argparse
import unittest

from vmm_benchmark_utils import MIB, byte_shape, describe, mib_bytes, split_mapped_bytes


class TestBenchmarkSizes(unittest.TestCase):
    def test_fractional_mib_preserves_exact_bytes(self):
        for text, expected in [("3", 3 * MIB), ("7", 7 * MIB), ("7.5", 15 * MIB // 2)]:
            with self.subTest(size=text):
                self.assertEqual(mib_bytes(text), expected)
                shape = byte_shape(expected)
                elements = 1
                for dimension in shape:
                    elements *= dimension
                self.assertEqual(elements, expected)
        self.assertEqual(byte_shape(3), (3,))

    def test_rejects_invalid_or_partial_byte_sizes(self):
        for text in ["0", "-3", "NaN", "Infinity", "bad", "0.0000001"]:
            with self.subTest(size=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    mib_bytes(text)

    def test_non_aligned_halves_report_padding_and_actual_ld_boundary(self):
        block = 2 * MIB
        for text, mapped_mib, boundary_mib, domain1_bytes in [
            ("3", 4, 2, MIB),
            ("7", 8, 4, 3 * MIB),
            ("7.5", 8, 4, 7 * MIB // 2),
        ]:
            with self.subTest(size=text):
                logical = mib_bytes(text)
                mapped = split_mapped_bytes(logical, block)
                self.assertEqual(mapped, mapped_mib * MIB)
                pages = [
                    {"domain": int(offset >= mapped // 2), "handle": index, "offset": offset}
                    for index, offset in enumerate(range(0, mapped, block))
                ]
                info = describe({"base": 4096, "logical_bytes": logical,
                                 "reserved_bytes": 20 * MIB, "block_bytes": block, "pages": pages})
                self.assertEqual(info["physical_domain_boundary_bytes"], boundary_mib * MIB)
                self.assertEqual(info["padding_bytes"], mapped - logical)
                self.assertEqual(info["logical_bytes_by_domain"]["1"], domain1_bytes)
                self.assertEqual(sum(info["logical_bytes_by_domain"].values()), logical)


if __name__ == "__main__":
    unittest.main()
