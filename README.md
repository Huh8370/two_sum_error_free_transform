# two_sum_error_free_transform

Split a floating-point sum into its rounded result and the exact residual error using Knuth's error-free transformation.

```python
from two_sum_error_free_transform import two_sum

s, e = two_sum(0.1, 0.2)
print(s)  # 0.30000000000000004
print(e)  # -2.7755575615628914e-17
print(s + e == 0.1 + 0.2)  # True, exactly
```

The function returns a `(sum, error)` tuple. For finite inputs the identity
`a + b == s + e` holds exactly in real arithmetic; the error term carries the
low-order bits that the rounded sum discarded.

## Why this exists

Adding two floats with very different magnitudes silently drops bits:
`1e16 + 1.0` rounds to `1e16`, losing the `1.0` entirely. The error-free
transformation recovers that lost part as a second float, which is the
building block for compensated summation, exact geometric predicates, and
interval arithmetic. The trade-off is one extra subtraction and one extra
addition per call; the Fast2Sum shortcut is used when the larger operand is
known first, saving one operation while remaining exact.

## Edge cases

Non-finite inputs are handled explicitly: if either operand is infinite or
NaN, the returned error is `0.0` and the sum follows IEEE-754 propagation
rules (`inf + -inf` yields NaN). This is a deliberate choice — there is no
meaningful low-order residual to recover from a non-finite sum, and returning
`0.0` keeps the result usable in further arithmetic. Signed zeros are treated
as ordinary finite values, so `two_sum(0.0, 0.0)` returns `(0.0, 0.0)` with a
positive-signed sum.
