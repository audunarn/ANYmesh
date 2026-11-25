"""The native exact predicates need separately rounded floating-point operations.

A fused multiply-add contraction (clang's default on arm64; MSVC with
``/fp:contract``) breaks Dekker's ``split`` and with it ``two_product`` and every
exact orientation/in-circle expansion.  ``setup.py`` builds the extension with
``-ffp-contract=off`` on non-Windows platforms; this test checks the compiled
transforms against exact rational arithmetic, so it fails on any platform where
the extension was compiled with contraction.
"""
from fractions import Fraction
import math

import numpy as np
import pytest


def _probe():
    extension = pytest.importorskip("anymesher._native")
    probe = getattr(extension, "exact_arithmetic_probe", None)
    assert callable(probe), "rebuild the extension: this checkout requires exact_arithmetic_probe"
    return probe


def _pairs():
    rng = np.random.default_rng(20261125)
    for _ in range(4000):
        a = float(rng.uniform(-1., 1.) * 2. ** int(rng.integers(-40, 40)))
        b = float(rng.uniform(-1., 1.) * 2. ** int(rng.integers(-40, 40)))
        yield a, b
    # Near-collinear station coordinates, where cancellation matters most.
    for k in range(200):
        a = 3.7 + 4. * k / 199. * math.cos(.3)
        yield a, a * math.tan(.3) + 1e-12 * k


def _sign(value):
    return (value > 0) - (value < 0)


def test_error_free_transforms_are_exact():
    probe = _probe()
    failures = []
    for a, b in _pairs():
        product, product_tail, total, total_tail, high, low, orientation = probe(a, b)
        A, B = Fraction(a), Fraction(b)
        if Fraction(product) + Fraction(product_tail) != A * B:
            failures.append(("two_product", a, b))
        if Fraction(total) + Fraction(total_tail) != A + B:
            failures.append(("two_sum", a, b))
        if Fraction(high) + Fraction(low) != A or (
                high != 0. and (int(Fraction(high) / 2. ** (math.frexp(high)[1] - 26)).bit_length() > 26
                                or Fraction(high) / 2. ** (math.frexp(high)[1] - 26) % 1 != 0)):
            failures.append(("split", a, b))
        p, q, r = (A, B), (B, A), (Fraction(a + b), Fraction(a - b))
        exact = _sign((p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0]))
        if orientation != exact:
            failures.append(("orientation", a, b))
    assert not failures, f"{len(failures)} inexact results (fused multiply-add?), first: {failures[:5]}"
