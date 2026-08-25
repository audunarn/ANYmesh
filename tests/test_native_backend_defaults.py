from __future__ import annotations

from inspect import signature
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

import anymesher
import anymesher.hybrid as hybrid_module
import anymesher.native_cpp as native_cpp
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.native import (
    NativeTriangulation,
    clear_native_boundary,
    register_native_boundary,
)
from anymesher.surface_mesh import SurfaceMeshOptions, mesh_planar_surface
from anymesher.structural_pipeline import OverlapPolicy
from anymesher.triangulation import (
    constrained_planar_triangulation,
    triangulate_polygon,
)


SQUARE = np.array(
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    dtype=np.float64,
)
OUTER = (0, 1, 2, 3)


class ProbeBoundary:
    name = "default-probe"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def triangulate(self, points, segments, outer_loop, hole_loops):
        self.calls += 1
        if self.fail:
            raise AssertionError("explicit Python invoked a native boundary")
        return NativeTriangulation(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        )


@pytest.fixture
def isolated_native_boundary():
    previous = clear_native_boundary()
    try:
        yield
    finally:
        register_native_boundary(previous)


def test_all_six_public_native_backend_defaults_are_auto(monkeypatch) -> None:
    assert signature(generate_hybrid_mesh_result).parameters[
        "native_backend"
    ].default == "auto"
    assert signature(SurfaceMeshOptions).parameters["backend"].default == "auto"
    assert SurfaceMeshOptions().backend == "auto"
    assert signature(mesh_planar_surface).parameters["backend"].default == "auto"
    assert signature(constrained_planar_triangulation).parameters[
        "backend"
    ].default == "auto"
    assert signature(triangulate_polygon).parameters["backend"].default == "auto"

    captured = {}
    expected_mesh = object()

    def fake_result(_geometry, **options):
        captured.update(options)
        return SimpleNamespace(mesh=expected_mesh)

    monkeypatch.setattr(hybrid_module, "generate_hybrid_mesh_result", fake_result)
    assert hybrid_module.generate_hybrid_mesh(object(), target_size=1.0) is expected_mesh
    assert captured == {"target_size": 1.0}


def test_hybrid_default_accepts_only_explicitly_declared_overlap_relations() -> None:
    assert signature(generate_hybrid_mesh_result).parameters[
        "overlap_policy"
    ].default is OverlapPolicy.CONNECT_DECLARED


def test_omitted_backend_selects_auto_and_records_provenance(
    isolated_native_boundary,
) -> None:
    boundary = ProbeBoundary()
    register_native_boundary(boundary)

    result = constrained_planar_triangulation(SQUARE, OUTER)

    assert result.requested_backend == "auto"
    assert result.selected_backend == boundary.name
    assert result.actual_backend == boundary.name
    assert result.fallback_reason is None
    assert boundary.calls == 1


def test_explicit_python_never_invokes_available_native_boundary(
    isolated_native_boundary,
) -> None:
    boundary = ProbeBoundary(fail=True)
    register_native_boundary(boundary)

    result = constrained_planar_triangulation(SQUARE, OUTER, backend="python")

    assert result.requested_backend == "python"
    assert result.selected_backend == "python"
    assert result.actual_backend == "python"
    assert result.fallback_reason is None
    assert boundary.calls == 0


def test_auto_falls_back_only_for_absent_native_capability(
    isolated_native_boundary, monkeypatch
) -> None:
    monkeypatch.setattr(native_cpp, "compiled_native_boundary", lambda: None)

    result = constrained_planar_triangulation(SQUARE, OUTER)

    assert result.requested_backend == "auto"
    assert result.selected_backend == "python"
    assert result.actual_backend == "python"
    assert result.fallback_reason == "native_capability_absent"
    with pytest.raises(MeshError, match="no native triangulation boundary"):
        constrained_planar_triangulation(SQUARE, OUTER, backend="native")


def test_auto_propagates_corrupt_native_boundary_errors(
    isolated_native_boundary, monkeypatch
) -> None:
    def corrupt_boundary():
        raise ImportError("compiled triangulation ABI is corrupt")

    monkeypatch.setattr(native_cpp, "compiled_native_boundary", corrupt_boundary)

    with pytest.raises(ImportError, match="ABI is corrupt"):
        constrained_planar_triangulation(SQUARE, OUTER)


def test_default_migration_remains_released_in_0_3_0() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert anymesher.__version__ == "0.3.0"
    assert project["project"]["version"] == "0.3.0"
