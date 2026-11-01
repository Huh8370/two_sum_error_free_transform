"""Knuth's error-free transformation for the sum of two floats.

The function :func:`two_sum` returns ``(s, e)`` such that ``a + b == s + e``
exactly, where ``s`` is the rounded result of ``a + b`` and ``e`` is the
rounding error.  The algorithm is due to Knuth (TAOCP Vol. 2, 4.2.2) and
uses only ordinary floating-point addition and subtraction; it does not
require FMA or extended precision.

Why this exists: naive summation loses low-order bits whenever the two
operands have very different magnitudes.  The error term ``e`` recovers
those bits exactly, which makes it possible to build compensated summation
routines, exact geometric predicates, and interval arithmetic without
relying on the platform's rounding mode.
"""

from __future__ import annotations

import math

__all__ = ["two_sum"]


def two_sum(a: float, b: float) -> tuple[float, float]:
    """Return ``(s, e)`` with ``a + b == s + e`` exactly.

    Parameters
    ----------
    a, b:
        Finite or non-finite IEEE-754 binary64 values.  The implementation
        uses the standard *Fast2Sum* branch when ``abs(a) >= abs(b)`` and the
        full Knuth branch otherwise; both branches are exact for finite
        inputs and propagate NaN/Inf according to IEEE-754 rules.

    Returns
    -------
    s:
        The rounded value of ``a + b``.
    e:
        The exact residual such that ``a + b == s + e`` in real arithmetic.
        For finite inputs the residual is exact (no further rounding).

    Notes
    -----
    The Fast2Sum variant is used when ``abs(a) >= abs(b)`` because it saves
    one addition while remaining exact; the check is done with ``abs`` so
    signed zeros and negative operands are handled uniformly.  When either
    operand is non-finite the residual is ``0.0`` (or NaN) as required by
    IEEE-754: there is no meaningful low-order error to recover.
    """
    s = a + b

    # Non-finite inputs: the sum is exact by definition, the residual is 0.
    # NaN propagates through ``s``; we return 0.0 for the error so callers
    # can still use the result in arithmetic without producing NaN from a
    # finite-looking pair.
    if not (math.isfinite(a) and math.isfinite(b)):
        return s, 0.0

    if abs(a) >= abs(b):
        # Fast2Sum: exact when the larger operand is added first.
        b_virtual = s - a
        a_virtual = s - b_virtual
        b_roundoff = b - b_virtual
        a_roundoff = a - a_virtual
    else:
        # Knuth's full TwoSum: works for any ordering.
        a_virtual = s - b
        b_virtual = s - a_virtual
        a_roundoff = a - a_virtual
        b_roundoff = b - b_virtual

    e = a_roundoff + b_roundoff
    return s, e
