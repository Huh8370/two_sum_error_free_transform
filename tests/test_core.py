"""Tests for :mod:`two_sum_error_free_transform`.

All assertions are deterministic: inputs are literal floats and the
comparisons use exact equality on values that are representable, or a
small tolerance where the exact bit pattern is not asserted.
"""

from __future__ import annotations

import math
import unittest

from two_sum_error_free_transform import two_sum


class TwoSumTests(unittest.TestCase):
    def assert_exact_sum(self, a: float, b: float) -> None:
        s, e = two_sum(a, b)
        self.assertEqual(s, a + b)
        # The defining property: a + b == s + e exactly.
        # Use exact equality because the transformation guarantees it.
        self.assertEqual(a + b, s + e)

    def test_simple_integers(self) -> None:
        s, e = two_sum(1.0, 2.0)
        self.assertEqual(s, 3.0)
        self.assertEqual(e, 0.0)

    def test_known_residual(self) -> None:
        # 0.1 + 0.2 == 0.30000000000000004 in binary64; the residual is the
        # low-order part that the rounded sum drops.
        s, e = two_sum(0.1, 0.2)
        self.assertEqual(s, 0.30000000000000004)
        self.assertNotEqual(e, 0.0)
        self.assertEqual(0.1 + 0.2, s + e)

    def test_large_and_small(self) -> None:
        a = 1.0e16
        b = 1.0
        s, e = two_sum(a, b)
        self.assertEqual(s, a + b)
        self.assertEqual(a + b, s + e)

    def test_negative_operands(self) -> None:
        self.assert_exact_sum(-1.5, -2.25)
        self.assert_exact_sum(-1.5, 2.25)
        self.assert_exact_sum(1.5, -2.25)

    def test_cancellation(self) -> None:
        # a and b nearly cancel; the residual carries the lost bits.
        a = 1.0000000000000002
        b = -1.0
        self.assert_exact_sum(a, b)

    def test_signed_zero(self) -> None:
        s, e = two_sum(0.0, 0.0)
        self.assertEqual(s, 0.0)
        self.assertEqual(e, 0.0)
        self.assertEqual(math.copysign(1.0, s), 1.0)

    def test_zero_plus_value(self) -> None:
        self.assert_exact_sum(0.0, 5.0)
        self.assert_exact_sum(-0.0, 5.0)
        self.assert_exact_sum(5.0, -0.0)

    def test_infinities(self) -> None:
        s, e = two_sum(math.inf, 1.0)
        self.assertEqual(s, math.inf)
        self.assertEqual(e, 0.0)

        s, e = two_sum(1.0, -math.inf)
        self.assertEqual(s, -math.inf)
        self.assertEqual(e, 0.0)

    def test_inf_minus_inf_is_nan(self) -> None:
        s, e = two_sum(math.inf, -math.inf)
        self.assertTrue(math.isnan(s))
        self.assertEqual(e, 0.0)

    def test_nan_propagates(self) -> None:
        s, e = two_sum(math.nan, 1.0)
        self.assertTrue(math.isnan(s))
        self.assertEqual(e, 0.0)

        s, e = two_sum(1.0, math.nan)
        self.assertTrue(math.isnan(s))
        self.assertEqual(e, 0.0)

    def test_both_branches_agree(self) -> None:
        # Exercise both the Fast2Sum branch (|a| >= |b|) and the Knuth branch
        # (|a| < |b|) with the same mathematical operands.
        for a, b in [(3.0, 1.0), (1.0, 3.0), (-3.0, 1.0), (1.0, -3.0)]:
            with self.subTest(a=a, b=b):
                self.assert_exact_sum(a, b)

    def test_subnormal_inputs(self) -> None:
        tiny = 5e-324  # smallest positive subnormal
        s, e = two_sum(tiny, tiny)
        self.assertEqual(s, 1e-323)
        self.assertEqual(e, 0.0)
        self.assertEqual(tiny + tiny, s + e)

    def test_exact_sum_property_on_grid(self) -> None:
        # Sweep a small deterministic grid of magnitudes and confirm the
        # defining identity a + b == s + e holds exactly.
        values = [0.0, 1.0, -1.0, 0.5, -0.5, 1e-300, 1e300, 1e16, -1e16]
        for a in values:
            for b in values:
                with self.subTest(a=a, b=b):
                    s, e = two_sum(a, b)
                    self.assertEqual(a + b, s + e)


if __name__ == "__main__":
    unittest.main()
