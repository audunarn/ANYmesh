"""orient2d's fallback is exact for binary-float inputs (tripod heave-plate probe, 2026-11-24).

An 80-digit decimal context rounded the differences of stations near 1e-16 next to stations near 1,
so exactly collinear seed stations got a nonzero, sign-wrong orientation and the Python seed CDT
refused the tripod heave-plate ring piece at 0.6-0.8 m with five different errors.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from anymesher.triangulation import incircle, orient2d

COLLINEAR = ((0.0, 0.0), (0.6921856373743935, 5.075305255429295e-16), (1.384371274748787, 1.015061051085859e-15))


def _exact(a, b, c):
    a, b, c = [(Fraction(float(p[0])), Fraction(float(p[1]))) for p in (a, b, c)]
    return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])


def test_exactly_collinear_stations_have_zero_orientation():
    assert _exact(*COLLINEAR) == 0
    for order in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2)):
        assert orient2d(*(COLLINEAR[i] for i in order)) == 0.0


def test_fallback_matches_exact_rational_sign_and_rounding():
    rng = np.random.default_rng(20261124)
    base = np.array(COLLINEAR)
    for _ in range(2000):
        pts = base + rng.integers(-3, 4, size=(3, 2)) * np.array((1e-17, 1e-31))
        pts[:, 0] *= rng.choice((1.0, 3.0, 1e-3))
        exact = _exact(*pts)
        value = orient2d(*pts)
        assert np.sign(value) == np.sign(float(exact)) == (exact > 0) - (exact < 0)
        assert value == float(exact)


def test_nonfinite_operands_keep_earlier_behaviour():
    assert np.isnan(orient2d((float("nan"), 0.0), (1.0, 0.0), (0.0, 1.0)))


def _ring_piece():
    from anygeometry.exact_curves import EllipticArc
    from anygeometry.model import GeometryModel
    from anygeometry.surfaces import Plane

    data = json.loads((Path(__file__).parent / "data" / "tripod_heave_plate_ring_piece_20261124.json").read_text())
    g, vertex, edges = GeometryModel(), {}, []

    def point(vid, xyz):
        if vid not in vertex:
            vertex[vid] = g.add_point(*xyz)
        return vertex[vid]

    for e in data["edges"]:
        a, b, c = point(e["start_id"], e["start"]), point(e["end_id"], e["end"]), e["curve"]
        edges.append(g.add_curve(a, b, EllipticArc(tuple(c["center"]), tuple(c["u"]), tuple(c["v"]), c["start"], c["sweep"]))
                     if c["kind"] == "elliptic" else g.add_line(a, b))
    p = data["plane"]
    g.add_face(edges, surface=Plane(tuple(p["origin"]), tuple(p["u"]), tuple(p["v"])))
    return g


@pytest.mark.parametrize("size", (0.6, 0.65, 0.7, 0.75, 0.8))
def test_tripod_heave_plate_ring_piece_meshes_quad_first(size):
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.quad.options import QuadMeshingOptions

    result = generate_hybrid_mesh_result(_ring_piece(), target_size=size, strategy="native", layout_policy="adaptive",
                                         quad_options=QuadMeshingOptions(quality_model="shape_jacobian"))
    assert len(result.mesh.quads) > 5 * len(result.mesh.tris) > 0


def test_underflow_regime_takes_the_exact_path():
    a = (2.0 ** -500, (1 + 2.0 ** -52) * 2.0 ** -575)
    b = ((1 - 2.0 ** -53) * 2.0 ** -500, 2.0 ** -575)
    c = (-float.fromhex("0x1.0000000000001p-553"), float.fromhex("0x1.7ffffffffffffp-628"))
    exact = _exact(a, b, c)
    assert exact < 0 and orient2d(a, b, c) < 0.0


def test_a_nonzero_determinant_below_the_subnormal_range_keeps_its_sign():
    tiny = 2.0 ** -600
    assert _exact((0.0, 0.0), (tiny, 0.0), (0.0, tiny)) > 0
    assert orient2d((0.0, 0.0), (tiny, 0.0), (0.0, tiny)) > 0.0
    assert orient2d((0.0, 0.0), (0.0, tiny), (tiny, 0.0)) < 0.0


def test_exact_zero_keeps_the_earlier_signed_zero():
    value = orient2d((-0.0, -0.0), (-0.0, 0.0), (0.0, 0.0))
    assert value == 0.0 and str(value) == "-0.0"
    assert str(orient2d(*COLLINEAR)) in ("0.0", "-0.0")


def test_vectorised_screen_matches_the_scalar_oracle_in_the_underflow_regime():
    from anymesher.native_cpp import _orientation_rows

    rng = np.random.default_rng(7)
    points = np.vstack([rng.normal(size=(12, 2)) * scale for scale in (1.0, 2.0 ** -500, 2.0 ** -540)]
                       + [np.array(COLLINEAR)])
    rows = np.array([(i, j, k) for i in range(len(points)) for j in range(len(points)) for k in range(len(points))
                     if len({i, j, k}) == 3][::7], dtype=np.int64)
    values = _orientation_rows(points, rows, orient2d)
    for (i, j, k), value in zip(rows, values):
        expected = orient2d(points[i], points[j], points[k])
        assert value == expected or (np.isnan(value) and np.isnan(expected))
        exact = _exact(points[i], points[j], points[k])
        assert np.sign(value) == (exact > 0) - (exact < 0)


def _exact_incircle(a, b, c, p):
    (ax, ay), (bx, by), (cx, cy), (px, py) = [(Fraction(float(q[0])), Fraction(float(q[1]))) for q in (a, b, c, p)]
    ax, ay, bx, by, cx, cy = ax - px, ay - py, bx - px, by - py, cx - px, cy - py
    value = (ax * ax + ay * ay) * (bx * cy - by * cx) - (bx * bx + by * by) * (ax * cy - ay * cx) \
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    orientation = _exact(a, b, c)
    return -value if orientation < 0 else value


def _sign(value):
    return (value > 0) - (value < 0)


def test_incircle_sign_is_exact_for_tiny_and_mixed_scale_inputs():
    s = 2.0 ** -300
    cases = [((0.0, 0.0), (s, 0.0), (s, s), (0.0, s / 2)),
             ((2.0 ** -500, (1 + 2.0 ** -52) * 2.0 ** -575), ((1 - 2.0 ** -53) * 2.0 ** -500, 2.0 ** -575),
              (2.0 ** 500, 0.0), (-float.fromhex("0x1.0000000000001p-553"), float.fromhex("0x1.7ffffffffffffp-628")))]
    rng = np.random.default_rng(11)
    for scale in (2.0 ** -520, 2.0 ** -300, 2.0 ** 300):
        for _ in range(200):
            cases.append(tuple(tuple(map(float, row)) for row in rng.normal(size=(4, 2)) * scale))
    for a, b, c, p in cases:
        assert _sign(incircle(a, b, c, p)) == _sign(_exact_incircle(a, b, c, p)), (a, b, c, p)


def test_filters_do_not_raise_under_strict_numpy_error_settings():
    from anymesher.native_cpp import _orientation_rows

    s = 2.0 ** -530
    points = np.array([(0.0, 0.0), (s, 0.0), (0.0, s)])
    with np.errstate(all="raise"):
        values = _orientation_rows(points, np.array([(0, 1, 2)], dtype=np.int64), orient2d)
        assert values[0] > 0.0 and orient2d(*points) > 0.0
        assert incircle((0.0, 0.0), (s, 0.0), (s, s), (0.0, s / 2)) > 0.0


def test_collinear_inputs_near_overflow_give_a_signed_zero_not_nan():
    from anymesher.native_cpp import _orientation_rows

    m = 2.0 ** 1023
    points = np.array([(m, 0.0), (0.0, 0.0), (-m, 0.0)])
    value = orient2d(*points)
    assert value == 0.0 and not np.isnan(value)
    assert _orientation_rows(points, np.array([(0, 1, 2)], dtype=np.int64), orient2d)[0] == 0.0


def test_ordinary_ambiguous_value_is_bit_identical_to_the_earlier_path():
    x = 0.25 - 2.0 ** -54
    a, b, c = (x, x), (0.5 + 2.0 ** -51, 0.5), (1.0, 1.0)
    assert _exact(a, b, c) > 0
    assert orient2d(a, b, c) == float.fromhex("0x1.8000000000001p-52")


def test_a_fallback_the_decimal_evaluates_exactly_does_no_integer_work(monkeypatch):
    """Cost guard (48-web paired timing, 2026-11-25): about 28 % of orient2d calls fall back, mostly exactly
    collinear stations whose 80-digit decimal value is exact.  That value is the earlier one and its sign is
    exact, so it is returned directly; only an inexact decimal evaluation takes the integer path."""
    import anymesher.triangulation as triangulation

    def integer_work(*_args, **_kwargs):
        raise AssertionError("integer path taken for a decimal-exact fallback")

    monkeypatch.setattr(triangulation, "_signed_zero_orient2d", integer_work)
    monkeypatch.setattr(triangulation, "_integer_orient2d", integer_work, raising=False)
    for points in (((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)), ((0.5, 0.25), (1.5, 0.75), (2.5, 1.25)),
                   ((-0.0, 0.0), (0.0, 1.5), (0.0, -3.25))):
        value = orient2d(*points)
        assert value == float(Fraction(*_exact_ratio(points))), points


def _exact_ratio(points):
    (ax, ay), (bx, by), (cx, cy) = ((Fraction(x), Fraction(y)) for x, y in points)
    exact = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return exact.numerator, exact.denominator


def test_an_inexact_decimal_fallback_with_a_certain_sign_keeps_the_earlier_value_without_integer_work(monkeypatch):
    """Nearly collinear real-geometry stations have long exact decimals, so the 80-digit evaluation rounds; when
    its value exceeds the rounding-error bound by the margin, it is returned (the earlier value, exact sign)."""
    import math
    from decimal import localcontext

    import anymesher.triangulation as triangulation

    def integer_work(*_args, **_kwargs):
        raise AssertionError("integer path taken although the decimal sign is certain")

    monkeypatch.setattr(triangulation, "_integer_orient2d", integer_work)
    points = ((0.1, 0.1), (0.3, math.nextafter(0.3, 1.0)), (0.0, 0.0))
    with localcontext() as context:
        context.prec = 80
        left, right = triangulation._decimal_orient2d_products(tuple(v for point in points for v in point))
        assert context.flags[triangulation.Inexact]  # the decimal evaluation rounds
    earlier = float(left - right)
    value = orient2d(*points)
    assert value == earlier and value > 0.0 and Fraction(*_exact_ratio(points)) > 0


def test_a_narrow_default_decimal_exponent_range_cannot_change_the_sign(monkeypatch):
    """Codex r5 (High): the private 80-digit context must not inherit DefaultContext's exponent range."""
    import decimal

    import anymesher.triangulation as triangulation

    monkeypatch.setattr(decimal.DefaultContext, "Emin", -170)
    monkeypatch.setattr(triangulation, "_DECIMAL_CONTEXT", triangulation._private_decimal_context())
    points = ((4.9e-250, 1.0), (5.1e-250, 2.0), (0.0, 0.0))
    assert Fraction(*_exact_ratio(points)) > 0
    assert orient2d(*points) == float.fromhex("0x1.aebaa746bbf7ep-829")


def test_a_caller_float_operation_trap_cannot_change_the_value():
    """Codex r5 (Medium): float -> Decimal conversion must not consult the caller's FloatOperation trap."""
    from decimal import FloatOperation, localcontext

    points = ((3 * 2.0 ** -809, 0.0), (0.0, 1 - 2.0 ** -52), (0.0, 0.0))
    expected = float.fromhex("0x1.7ffffffffffffp-808")
    assert orient2d(*points) == expected
    with localcontext() as context:
        context.traps[FloatOperation] = True
        assert orient2d(*points) == expected
        assert not context.flags[FloatOperation]
