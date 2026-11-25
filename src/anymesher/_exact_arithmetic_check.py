"""Exact rational check of the compiled error-free transforms.

Used by ``tests/test_exact_arithmetic_contraction.py`` and the release wheel
smoke.  Returns a list of failures (empty when every result is exact).
"""
from __future__ import annotations

import math
import random
from fractions import Fraction


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _pairs(count: int):
    rng = random.Random(20261125)
    for _ in range(count):
        yield (rng.uniform(-1., 1.) * 2. ** rng.randint(-40, 40),
               rng.uniform(-1., 1.) * 2. ** rng.randint(-40, 40))
    # Exactly collinear constructions (a == b, b == 0, b == -a), one-ULP
    # perturbations of them and a tiny b, inside the admitted 2**-64..2**64 range
    # (subnormal products underflow by nature and are outside it).
    for _ in range(200):
        a = rng.uniform(.5, 4.) * 2. ** rng.randint(-20, 20)
        for b in (a, 0., -a, math.nextafter(a, math.inf), math.nextafter(a, -math.inf),
                  a * 2. ** -40, math.nextafter(-a, math.inf)):
            yield a, b


def exact_arithmetic_failures(probe, count: int = 4000) -> list[tuple[str, float, float]]:
    failures = []
    for a, b in _pairs(count):
        product, product_tail, total, total_tail, high, low, orientation = probe(a, b)
        A, B = Fraction(a), Fraction(b)
        if Fraction(product) + Fraction(product_tail) != A * B:
            failures.append(("two_product", a, b))
        if Fraction(total) + Fraction(total_tail) != A + B:
            failures.append(("two_sum", a, b))
        # Dekker split: a == high + low with high carrying at most 26 bits.
        if Fraction(high) + Fraction(low) != A or (
                high != 0. and (Fraction(high) / Fraction(2) ** (math.frexp(high)[1] - 26)).denominator != 1):
            failures.append(("split", a, b))
        p, q, r = (A, B), (B, A), (Fraction(a + b), Fraction(a - b))
        exact = _sign((p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0]))
        if orientation != exact:
            failures.append(("orientation", a, b))
    return failures
