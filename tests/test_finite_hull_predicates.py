"""Exact sign regressions for near-collinear finite-hull flip decisions."""
from fractions import Fraction
from itertools import permutations

import pytest

from anymesher.triangulation import incircle


# Four original stations from a plate rotated by 0.37 radians. The previous
# floating-point filter accepted inconsistent signs and cycled between two
# finite-hull triangulations. Preserve the actual binary input coordinates.
QUAD = (
    (0.23308183640150862, 0.0904038579912405),
    (1.1654091820075432, 0.4520192899562025),
    (0.46616367280301724, 0.180807715982481),
    (0.6992455092045259, 0.2712115739737215),
)


def exact_sign(a, b, c, d):
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = (
        tuple(Fraction(value) for value in row) for row in (a, b, c, d)
    )
    orientation = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)
    ax, ay, bx, by, cx, cy = ax-dx, ay-dy, bx-dx, by-dy, cx-dx, cy-dy
    value = ((ax*ax+ay*ay)*(bx*cy-by*cx)
             - (bx*bx+by*by)*(ax*cy-ay*cx)
             + (cx*cx+cy*cy)*(ax*by-ay*bx))
    if orientation < 0:
        value = -value
    return (value > 0) - (value < 0)


@pytest.mark.parametrize('indices', tuple(permutations(range(4))))
def test_near_collinear_incircle_sign_matches_exact_binary_input(indices):
    points = tuple(QUAD[index] for index in indices)
    result = incircle(*points)
    assert int(result > 0) - int(result < 0) == exact_sign(*points)
