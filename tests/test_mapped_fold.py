"""Mapped (transfinite) meshing must never publish a folded grid.

ANYgeometry can declare four corners on an L-shaped outline, putting the
re-entrant corner inside one mapped side.  Blending those sides folds the grid
over: through 0.5.0 the automatic strategy published such a mesh with a
quarter of its nodes outside the face and element areas summing to more than
the face area.  Automatic meshing now sends such faces to native meshing, the
explicit mapped strategy refuses them, and native meshing of a Plane face uses
the plane's exact chart instead of the clipped four-corner patch.
"""
from __future__ import annotations

import numpy as np
import pytest

from anygeometry import GeometryModel

from anymesher._mapped_fold import grid_folds, mapped_face_folds
from anymesher.decomposition import check_mappable
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result

L_SHAPE = ((0.0, 0.0), (4.0, 0.0), (4.0, 1.2), (2.2, 1.2), (2.2, 2.6), (0.0, 2.6))
L_MIRRORED = ((0.0, 0.0), (2.6, 0.0), (2.6, 2.2), (1.2, 2.2), (1.2, 4.0), (0.0, 4.0))
RECTANGLE = ((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0))
PENTAGON = ((0.0, 0.0), (5.0, 0.8), (6.0, 4.0), (2.5, 6.0), (-0.5, 3.5))


def _plate(points):
    model = GeometryModel()
    vertices = model.add_points(tuple((x, y, 0.0) for x, y in points))
    return model, model.add_plate(vertices)


def _surfaceless(points):
    model = GeometryModel()
    vertices = model.add_points(tuple((x, y, 0.0) for x, y in points))
    return model, model.add_face(model.add_polyline(vertices, close=True), surface=None)


def _polygon_area(points) -> float:
    xy = np.asarray(points, dtype=float)
    return 0.5 * float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))


def _assert_exact_cover(mesh, outline) -> None:
    """Every element positively oriented and areas summing to the face area.

    Together with the exact boundary this rules out folds and overlaps.
    """
    total = 0.0
    for body in (*(b[:4] for b in mesh.quads.values()), *(b[:3] for b in mesh.tris.values())):
        corners = [np.asarray(mesh.nodes[node], dtype=float)[:2] for node in body]
        count = len(corners)
        for index in range(count):
            here, after, before = corners[index], corners[(index + 1) % count], corners[index - 1]
            outgoing, incoming = after - here, before - here
            assert outgoing[0] * incoming[1] - outgoing[1] * incoming[0] > 0.0
        total += _polygon_area(corners)
    assert total == pytest.approx(_polygon_area(outline), rel=1e-12)


@pytest.mark.parametrize("outline", (L_SHAPE, L_MIRRORED))
def test_automatic_strategy_meshes_l_plate_natively_without_folding(outline) -> None:
    model, face = _plate(outline)
    assert mapped_face_folds(model, face)
    mesh = generate_hybrid_mesh_result(model, target_size=0.2, face_ids=(face,)).mesh
    assert mesh.hybrid_diagnostics["strategy_by_face"] == {face: "native"}
    _assert_exact_cover(mesh, outline)


@pytest.mark.parametrize("outline", (L_SHAPE, L_MIRRORED))
def test_explicit_mapped_strategy_refuses_a_folding_face(outline) -> None:
    model, face = _plate(outline)
    with pytest.raises(MeshError, match="folds over"):
        generate_hybrid_mesh_result(model, target_size=0.2, face_ids=(face,), strategy="mapped")


def test_check_mappable_reports_the_fold() -> None:
    model, face = _plate(L_SHAPE)
    report = check_mappable(model, face)
    assert not report.ok
    assert any("folds over" in message for message in report.messages)
    rectangle, rectangle_face = _plate(RECTANGLE)
    assert check_mappable(rectangle, rectangle_face).ok


def test_surfaceless_l_face_fails_with_an_actionable_error() -> None:
    # Its only surface is the topology Coons blend, which itself folds.
    for strategy in ("auto", "native"):
        model, face = _surfaceless(L_SHAPE)
        with pytest.raises(MeshError, match="add_plate"):
            generate_hybrid_mesh_result(model, target_size=0.2, face_ids=(face,), strategy=strategy)


@pytest.mark.parametrize("outline", (RECTANGLE, PENTAGON))
@pytest.mark.parametrize("build", (_plate, _surfaceless))
def test_non_folding_four_sided_faces_stay_mapped(outline, build) -> None:
    model, face = build(outline)
    assert not mapped_face_folds(model, face)
    mesh = generate_hybrid_mesh_result(model, target_size=0.2, face_ids=(face,)).mesh
    assert mesh.hybrid_diagnostics["strategy_by_face"] == {face: "mapped"}
    assert not mesh.tris
    _assert_exact_cover(mesh, outline)


def test_grid_folds_detects_inversion_but_accepts_curvature() -> None:
    u, v = np.meshgrid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 5), indexing="ij")
    flat = np.stack((u, v, np.zeros_like(u)), axis=-1)
    assert not grid_folds(flat)
    angle = u * (np.pi / 2.0)
    curved = np.stack((np.cos(angle), np.sin(angle), v), axis=-1)  # quarter cylinder
    assert not grid_folds(curved)
    folded = flat.copy()
    folded[2, 2, :2] = (0.9, 0.9)  # drag one interior station across its neighbours
    assert grid_folds(folded)
