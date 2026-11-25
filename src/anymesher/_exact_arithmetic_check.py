"""Exact rational check of the compiled error-free transforms.

Used by ``tests/test_exact_arithmetic_contraction.py`` and the release wheel
smoke.  Returns a list of failures (empty when every result is exact).
"""
from __future__ import annotations

import math
import random
import sys
from fractions import Fraction


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


_LOW, _HIGH = 2. ** -64, 2. ** 64


def _admitted(*values: float) -> bool:
    # The production coordinate range: zero or 2**-64 <= |v| <= 2**64.
    return all(v == 0. or _LOW <= abs(v) <= _HIGH for v in values)


def _pairs(count: int):
    rng = random.Random(20261125)
    for _ in range(count):
        a = rng.uniform(-1., 1.) * 2. ** rng.randint(-40, 40)
        b = rng.uniform(-1., 1.) * 2. ** rng.randint(-40, 40)
        if _admitted(a, b, a + b, a - b):
            yield a, b
    # Equal, opposite, zero and one-ULP neighbours, scaled so that every
    # derived coordinate stays inside the admitted range.
    for _ in range(200):
        a = rng.uniform(.5, 4.) * 2. ** rng.randint(-10, 20)
        for b in (a, 0., -a, math.nextafter(a, math.inf), math.nextafter(a, -math.inf),
                  a * 2. ** -40, math.nextafter(-a, math.inf)):
            if _admitted(a, b, a + b, a - b):
                yield a, b


def _triples(count: int):
    """Exactly collinear, well-scaled triples and one-ULP perturbations of the
    third point to either side, so both orientation signs and zero occur."""
    rng = random.Random(1125)
    for _ in range(count):
        scale = 2. ** rng.randint(-30, 30)
        x0, y0 = (rng.randint(-2 ** 20, 2 ** 20) * scale for _ in range(2))
        dx, dy = (rng.randint(-2 ** 20, 2 ** 20) * scale for _ in range(2))
        t = rng.randint(2, 9)
        p, q, r = (x0, y0), (x0 + dx, y0 + dy), (x0 + t * dx, y0 + t * dy)
        for third in (r, (math.nextafter(r[0], math.inf), r[1]), (math.nextafter(r[0], -math.inf), r[1]),
                      (r[0], math.nextafter(r[1], math.inf)), (r[0], math.nextafter(r[1], -math.inf))):
            if _admitted(*p, *q, *third):
                yield p, q, third


def _subnormals_preserved() -> bool:
    # A fast-math startup object linked into the extension can set
    # flush-to-zero / denormals-are-zero for the importing thread.
    tiny = sys.float_info.min
    half = tiny / 2.
    return half != 0. and half * 2. == tiny and math.ulp(0.) > 0.


def exact_arithmetic_failures(probe, count: int = 4000) -> list[tuple]:
    failures = []
    if not _subnormals_preserved():
        failures.append(("subnormal", "flush-to-zero after import", None))
    signs = set()
    for p, q, r in _triples(count // 4):
        orientation = probe(1., 1., *p, *q, *r)[6]
        P, Q, R = ([Fraction(c) for c in point] for point in (p, q, r))
        exact = _sign((P[0] - R[0]) * (Q[1] - R[1]) - (P[1] - R[1]) * (Q[0] - R[0]))
        signs.add(exact)
        if orientation != exact:
            failures.append(("orientation", p, q, r))
    if signs != {-1, 0, 1}:
        failures.append(("corpus", "orientation signs", tuple(sorted(signs))))
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
