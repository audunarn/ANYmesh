"""Bounded adaptive fallback and complex-geometry diagnostics."""

from __future__ import annotations

import pytest

from anygeometry import GeometryModel

from anymesher.backends import base
from anymesher.errors import MeshError
from anymesher.mesh import Mesh


def _rectangle() -> GeometryModel:
    model = GeometryModel()
    vertices = model.add_points(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    model.add_face(model.add_polyline(vertices, close=True))
    return model


def test_auto_uses_gmsh_only_for_qualified_planar_unshared_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _rectangle()
    calls: list[str] = []

    def reject_hybrid(_geometry: GeometryModel, **_options: object) -> Mesh:
        calls.append("hybrid")
        raise MeshError("native chart is unsupported")

    def accept_gmsh(_geometry: GeometryModel, **_options: object) -> Mesh:
        calls.append("gmsh")
        return Mesh()

    monkeypatch.setitem(base._BACKEND_LOADERS, "auto", lambda: reject_hybrid)
    monkeypatch.setattr(base, "_gmsh_backend", lambda: accept_gmsh)

    mesh = base.generate_mesh(geometry, target_size=0.25)

    assert calls == ["hybrid", "gmsh"]
    assert mesh.geometry_model_id == geometry.model_id
    assert mesh.geometry_revision == geometry.revision
    adaptive = mesh.hybrid_diagnostics["complex_geometry"]
    assert adaptive["selected_strategy"] == "gmsh"
    assert adaptive["attempted_strategies"] == ["hybrid", "gmsh"]


def test_explicit_native_never_uses_gmsh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _rectangle()
    calls: list[str] = []

    def reject_native(_geometry: GeometryModel, **_options: object) -> Mesh:
        calls.append("native")
        raise MeshError("native rejected")

    def forbidden_gmsh(_geometry: GeometryModel, **_options: object) -> Mesh:
        calls.append("gmsh")
        return Mesh()

    monkeypatch.setitem(base._BACKEND_LOADERS, "native", lambda: reject_native)
    monkeypatch.setattr(base, "_gmsh_backend", lambda: forbidden_gmsh)

    with pytest.raises(MeshError, match="native rejected"):
        base.generate_mesh(geometry, backend="native", target_size=0.25)

    assert calls == ["native"]


def test_auto_does_not_externalize_hybrid_only_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _rectangle()
    calls: list[str] = []

    def reject_hybrid(_geometry: GeometryModel, **_options: object) -> Mesh:
        calls.append("hybrid")
        raise MeshError("structured policy rejected")

    def forbidden_gmsh(_geometry: GeometryModel, **_options: object) -> Mesh:
        calls.append("gmsh")
        return Mesh()

    monkeypatch.setitem(base._BACKEND_LOADERS, "auto", lambda: reject_hybrid)
    monkeypatch.setattr(base, "_gmsh_backend", lambda: forbidden_gmsh)

    with pytest.raises(MeshError, match="structured policy rejected"):
        base.generate_mesh(
            geometry,
            target_size=0.25,
            structured_options={"preference": "balanced"},
        )

    assert calls == ["hybrid"]
