from __future__ import annotations

import numpy as np
import pytest

from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.high_order import ValidityStatus, certify_mapping_validity
from anymesher.quad.options import QuadMeshingOptions


def _face_id(model, face_use) -> int:
    return int(model.face_uses[face_use.id].face_id)


def _generate(model, face_ids, target_size, *, order="linear", cancellation_check=None):
    return generate_hybrid_mesh_result(
        model,
        face_ids=tuple(face_ids),
        target_size=float(target_size),
        strategy="native",
        native_backend="python",
        recombine=True,
        order=order,
        quad_options=QuadMeshingOptions(),
        cancellation_check=cancellation_check,
    )


def _corner_body(body: tuple[int, ...]) -> tuple[int, ...]:
    if len(body) in (4, 8):
        return tuple(body[:4])
    if len(body) in (3, 6):
        return tuple(body[:3])
    raise AssertionError(f"unexpected shell width {len(body)}")


def _edge_mid_map(mesh) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for body in list(mesh.quads.values()) + list(mesh.tris.values()):
        corners = _corner_body(tuple(body))
        midsides = tuple(body[len(corners):])
        assert len(midsides) == len(corners)
        for i, midpoint in enumerate(midsides):
            a, b = corners[i], corners[(i + 1) % len(corners)]
            key = (min(a, b), max(a, b))
            if key in result:
                assert result[key] == midpoint
            result[key] = midpoint
    return result


def _assert_strict_valid(mesh) -> None:
    for body in mesh.quads.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "Q8").status is ValidityStatus.CERTIFIED_POSITIVE
    for body in mesh.tris.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "T6").status is ValidityStatus.CERTIFIED_POSITIVE


def test_ch4_single_sector_promotes_same_linear_topology() -> None:
    linear_model, selected = _sector_model(False)
    linear_face = _face_id(linear_model, selected[0])
    linear = _generate(linear_model, (linear_face,), 0.5, order="linear").mesh
    model, selected = _sector_model(False)
    before = _persistent_state(model)
    face = _face_id(model, selected[0])
    quadratic = _generate(model, (face,), 0.5, order="quadratic").mesh
    assert len(quadratic.quads) == len(linear.quads) == 8
    assert len(quadratic.tris) == len(linear.tris) == 2
    assert set(quadratic.quads) == set(linear.quads)
    assert set(quadratic.tris) == set(linear.tris)
    for eid in quadratic.quads:
        assert tuple(quadratic.quads[eid][:4]) == tuple(linear.quads[eid])
    for eid in quadratic.tris:
        assert tuple(quadratic.tris[eid][:3]) == tuple(linear.tris[eid])
    for node, point in linear.nodes.items():
        np.testing.assert_allclose(quadratic.nodes[node], point, rtol=0.0, atol=0.0)
    midsides = _edge_mid_map(quadratic)
    assert len(quadratic.nodes) == len(linear.nodes) + len(midsides)
    assert set(midsides.values()) == set(quadratic.nodes) - set(linear.nodes)
    points = np.asarray([quadratic.nodes[node] for node in midsides.values()], dtype=float)
    np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 1.0, atol=1.0e-10)
    for edge_id, chain in linear.nodes_of_edge.items():
        assert tuple(quadratic.nodes_of_edge[edge_id][::2]) == tuple(chain)
    assert quadratic.order == "quadratic"
    _assert_strict_valid(quadratic)
    assert _persistent_state(model) == before


def test_ch4_fine_sector_preserves_ch3_corner_topology() -> None:
    linear_model, selected = _sector_model(False)
    linear_face = _face_id(linear_model, selected[0])
    linear = _generate(linear_model, (linear_face,), 0.25, order="linear").mesh
    model, selected = _sector_model(False)
    face = _face_id(model, selected[0])
    quadratic = _generate(model, (face,), 0.25, order="quadratic").mesh
    assert len(quadratic.quads) == len(linear.quads) == 30
    assert len(quadratic.tris) == len(linear.tris) == 2
    assert {e: tuple(b[:4]) for e, b in quadratic.quads.items()} == linear.quads
    assert {e: tuple(b[:3]) for e, b in quadratic.tris.items()} == linear.tris
    _assert_strict_valid(quadratic)


def test_ch4_full_ring_reuses_quadratic_periodic_seam_ids() -> None:
    model, selected = _sector_model(False)
    before = _persistent_state(model)
    faces = tuple(_face_id(model, use) for use in selected)
    seam = [edge for edge in model.edges if set(model.faces_using_edge(edge)) == {faces[0], faces[-1]}]
    assert len(seam) == 1
    result = _generate(model, faces, 0.5, order="quadratic")
    mesh = result.mesh
    chain = tuple(mesh.nodes_of_edge[int(seam[0])])
    assert len(chain) >= 3 and len(chain) % 2 == 1
    assert len(chain) == len(set(chain))
    midsides = chain[1::2]
    points = np.asarray([mesh.nodes[node] for node in midsides], dtype=float)
    np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 1.0, atol=1.0e-10)
    _assert_strict_valid(mesh)
    assert _persistent_state(model) == before
    repeat_model, repeat_selected = _sector_model(False)
    repeat_faces = tuple(_face_id(repeat_model, use) for use in repeat_selected)
    repeat = _generate(repeat_model, repeat_faces, 0.5, order="quadratic").mesh
    assert (len(repeat.nodes), len(repeat.quads), len(repeat.tris)) == (
        len(mesh.nodes), len(mesh.quads), len(mesh.tris)
    )


def test_ch4_quadratic_cancellation_is_atomic_for_source() -> None:
    model, selected = _sector_model(False)
    before = _persistent_state(model)
    face = _face_id(model, selected[0])

    class Cancelled(RuntimeError):
        pass

    def cancel(phase: str) -> None:
        if phase == "quad-first:quadratic-promotion-ready":
            raise Cancelled(phase)

    with pytest.raises(Cancelled):
        _generate(model, (face,), 0.5, order="quadratic", cancellation_check=cancel)
    assert _persistent_state(model) == before
