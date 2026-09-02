from __future__ import annotations

import numpy as np
import pytest

import anymesher.native_cpp as native_cpp
from anymesher.core import MeshCore
from anymesher.native_cpp import COMPILED_QUALITY_PIPELINE_AVAILABLE
from anymesher.quality_v2 import quad_quality, triangle_quality
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
