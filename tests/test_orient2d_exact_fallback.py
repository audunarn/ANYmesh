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

from anymesher.triangulation import orient2d

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
