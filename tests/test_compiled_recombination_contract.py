from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import pytest

import anymesher.native_cpp as native_cpp
from anymesher.core import MeshCore
from anymesher.native_cpp import COMPILED_QUALITY_PIPELINE_AVAILABLE
from anymesher.quality_v2 import quad_candidate_quality, quad_quality, triangle_quality
from anymesher.recombine import recombine_triangles_with_report


@pytest.mark.skipif(
    not COMPILED_QUALITY_PIPELINE_AVAILABLE,
    reason="compiled quality pipeline has not been rebuilt",
)
@pytest.mark.parametrize("reverse", (False, True))
def test_compiled_recombination_matches_python_reference(reverse: bool) -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(
        ((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4)),
        dtype=np.int64,
    )
    if reverse:
        triangles = np.ascontiguousarray(triangles[::-1])
    mesh = MeshCore(points, triangles)

    python = recombine_triangles_with_report(mesh, _use_native=False)
    compiled = recombine_triangles_with_report(mesh, _use_native=True)

    np.testing.assert_array_equal(
        compiled.paired_triangle_ids, python.paired_triangle_ids
    )
    np.testing.assert_array_equal(
        compiled.mesh.triangle_active, python.mesh.triangle_active
    )
    np.testing.assert_array_equal(
        compiled.mesh.quad_connectivity, python.mesh.quad_connectivity
    )
    np.testing.assert_allclose(compiled.scores, python.scores, rtol=1.0e-13, atol=1.0e-14)
    assert compiled.candidate_count == python.candidate_count
    assert compiled.rejected_candidate_count == python.rejected_candidate_count
    assert compiled.exchange_count == python.exchange_count
    assert compiled.exchange_work == python.exchange_work
    assert compiled.exchange_truncated is python.exchange_truncated


@pytest.mark.skipif(
    not COMPILED_QUALITY_PIPELINE_AVAILABLE,
    reason="compiled quality pipeline has not been rebuilt",
)
def test_compiled_recombination_preserves_protected_diagonal() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = MeshCore(points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64))

    report = recombine_triangles_with_report(
        mesh, protected_edges=((0, 2),), _use_native=True
    )

    assert report.pair_count == 0
    np.testing.assert_array_equal(report.mesh.triangle_active, (True, True))


@pytest.mark.skipif(
    not COMPILED_QUALITY_PIPELINE_AVAILABLE,
    reason="compiled quality pipeline has not been rebuilt",
)
def test_compiled_element_quality_matches_python_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.1),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    quads = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    monkeypatch.setattr(native_cpp, "COMPILED_QUALITY_PIPELINE_AVAILABLE", False)
    python_triangle = triangle_quality(points, triangles)
    python_quad = quad_quality(points, quads)
    monkeypatch.setattr(native_cpp, "COMPILED_QUALITY_PIPELINE_AVAILABLE", True)
    compiled_triangle = triangle_quality(points, triangles)
    compiled_quad = quad_quality(points, quads)

    for name in (
        "area",
        "aspect_ratio",
        "minimum_angle",
        "maximum_angle",
        "scaled_jacobian",
        "warpage",
    ):
        np.testing.assert_allclose(
            getattr(compiled_triangle, name),
            getattr(python_triangle, name),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            getattr(compiled_quad, name),
            getattr(python_quad, name),
            rtol=2.0e-14,
            atol=2.0e-14,
        )


# Original binary64 Q4 cycles from the cylindrical parity failure. Old acos
# measurements are deliberately not used as corrected reference values.
_CYLINDRICAL_WARPAGE_CASES = (
    (
        ("-0x1.6a09e667f3bccp-1", "0x1.6a09e667f3bcdp-1", "0x1.999999999999ap-2"),
        ("-0x1.6a09e667f3bccp-1", "0x1.6a09e667f3bcdp-1", "0x1.3333333333334p-1"),
        ("-0x1.3a1e1f22db17bp-1", "0x1.9451bd7525764p-1", "0x1.60db9ba7a234ep-1"),
        ("-0x1.3abca025d25a0p-1", "0x1.93d667187bce3p-1", "0x1.189fd66e42138p-1"),
    ),
    (
        ("0x1.1a62633145c07p-54", "0x1.0000000000000p+0", "0x1.8cccccccccccdp+0"),
        ("-0x1.b75140e67849dp-6", "0x1.ffd0df5859887p-1", "0x1.89e85a86b2de3p+0"),
        ("-0x1.6e872bcb208d0p-6", "0x1.ffdf328675abdp-1", "0x1.903c18004e8c1p+0"),
        ("0x1.1a62633145c07p-54", "0x1.0000000000000p+0", "0x1.9333333333334p+0"),
    ),
    (
        ("0x1.6a09e667f3bcbp-1", "-0x1.6a09e667f3bcep-1", "0x1.0000000000000p+0"),
        ("0x1.8b54efa9b4c0dp-1", "-0x1.455b3c8aa6d33p-1", "0x1.e80de8970fd40p-1"),
        ("0x1.8b56e75526d60p-1", "-0x1.4558d88809260p-1", "0x1.103b5a2e84f01p+0"),
        ("0x1.6a09e667f3bcbp-1", "-0x1.6a09e667f3bcep-1", "0x1.199999999999ap+0"),
    ),
)


@pytest.mark.skipif(
    not COMPILED_QUALITY_PIPELINE_AVAILABLE,
    reason="compiled quality pipeline has not been rebuilt",
)
@pytest.mark.parametrize("hex_points", _CYLINDRICAL_WARPAGE_CASES)
def test_cylindrical_near_planar_warpage_parity(hex_points) -> None:
    points = np.asarray([[float.fromhex(x) for x in row] for row in hex_points])
    connectivity = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    reference = quad_candidate_quality(points, connectivity[0])
    compiled = native_cpp.native_element_quality(points, connectivity, 4)
    assert compiled is not None
    np.testing.assert_allclose(
        compiled[0],
        [reference[name] for name in (
            "area", "aspect_ratio", "minimum_angle", "maximum_angle",
            "scaled_jacobian", "warpage",
        )],
        rtol=2.0e-14, atol=2.0e-14,
    )


@pytest.mark.parametrize("fraction", (0.0, 1.0e-12, 0.1 - 1.0e-10, 0.1, 0.1 + 1.0e-10, 0.5, 1.0))
def test_quad_warpage_matches_analytical_fold(fraction: float) -> None:
    angle = fraction * pi
    points = np.asarray(((0., 0., 0.), (0., -1., 0.), (1., 0., 0.),
                         (0., cos(angle), sin(angle))))
    reference = quad_candidate_quality(points, (0, 1, 2, 3))["warpage"]
    assert reference == pytest.approx(fraction, rel=2.0e-14, abs=2.0e-14)
    if fraction != 0.1:
        assert (reference <= 0.1) is (fraction <= 0.1)
    if COMPILED_QUALITY_PIPELINE_AVAILABLE:
        compiled = native_cpp.native_element_quality(
            points, np.asarray(((0, 1, 2, 3),), dtype=np.int64), 4)
        assert compiled is not None
        assert compiled[0, 5] == pytest.approx(reference, rel=2.0e-14, abs=2.0e-14)


@pytest.mark.parametrize("points", (
    ((0., 0., 0.), (0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
    ((0., 0., 0.), (0., -1., 0.), (1., 0., 0.), (0., 0., 0.)),
))
def test_quad_warpage_keeps_degenerate_normal_rejection(points) -> None:
    points = np.asarray(points)
    assert quad_candidate_quality(points, (0, 1, 2, 3))["warpage"] == 1.0
    if COMPILED_QUALITY_PIPELINE_AVAILABLE:
        compiled = native_cpp.native_element_quality(
            points, np.asarray(((0, 1, 2, 3),), dtype=np.int64), 4)
        assert compiled is not None
        assert compiled[0, 5] == 1.0
