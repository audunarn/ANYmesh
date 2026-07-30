"""Quality metrics, and the mesh JSON round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from anymesher import (
    ASPECT_RATIO_LIMIT,
    WARP_LIMIT,
    Coupling,
    GeometryModel,
    Mesh,
    MeshError,
    PanelMeshConfig,
    StiffenedPanel,
    generate_mesh,
    load_mesh,
    mesh_from_dict,
    mesh_to_dict,
    save_mesh,
    simple_panel_mesh,
    stiffened_panel_mesh,
    verify_mesh_quality,
)


def _one_quad(*corners) -> Mesh:
    mesh = Mesh()
    for index, position in enumerate(corners, start=1):
        mesh.nodes[index] = np.asarray(position, dtype=float)
    mesh.quads[1] = tuple(range(1, len(corners) + 1))
    return mesh


def test_a_square_is_perfect() -> None:
    quality = verify_mesh_quality(_one_quad((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))

    assert quality.num_shell_elements == 1
    assert quality.max_aspect_ratio == pytest.approx(1.0)
    assert quality.mean_aspect_ratio == pytest.approx(1.0)
    assert quality.max_warp == pytest.approx(0.0)
    assert quality.ok
    assert quality.warnings == ()


def test_a_stretched_element_is_reported_but_not_refused() -> None:
    quality = verify_mesh_quality(_one_quad((0, 0, 0), (10, 0, 0), (10, 1, 0), (0, 1, 0)))

    assert quality.max_aspect_ratio == pytest.approx(10.0)
    assert not quality.ok
    assert any("aspect ratio" in warning for warning in quality.warnings)
    # Advisory rather than enforced: the mesh still exists and is still returned.
    assert quality.num_shell_elements == 1


def test_a_warped_element_is_reported() -> None:
    quality = verify_mesh_quality(_one_quad((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0.5)))

    assert quality.max_warp > WARP_LIMIT
    assert any("warp" in warning for warning in quality.warnings)


def test_a_degenerate_element_does_not_divide_by_zero() -> None:
    quality = verify_mesh_quality(_one_quad((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))

    assert np.isfinite(quality.max_aspect_ratio)
    assert quality.max_warp == pytest.approx(0.0)


def test_an_empty_mesh_reports_neutral_values() -> None:
    quality = verify_mesh_quality(Mesh())

    assert quality.num_shell_elements == 0
    assert quality.max_aspect_ratio == pytest.approx(1.0)
    assert quality.ok


def test_quadratic_elements_are_measured_by_their_corners() -> None:
    linear = simple_panel_mesh(2.0, 1.0, 0.01, 4, 2, use_8node_elements=False)
    quadratic = simple_panel_mesh(2.0, 1.0, 0.01, 4, 2, use_8node_elements=True)

    # Including a mid-side node would halve the measured edge lengths and report
    # a quadratic mesh as a different shape from the linear one it matches.
    assert verify_mesh_quality(quadratic).max_aspect_ratio == pytest.approx(
        verify_mesh_quality(linear).max_aspect_ratio
    )


def test_triangles_count_as_shell_elements() -> None:
    mesh = Mesh()
    for index, position in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (4, 0, 0)], start=1):
        mesh.nodes[index] = np.asarray(position, dtype=float)
    mesh.tris[1] = (1, 2, 3)
    mesh.tris[2] = (1, 4, 3)

    quality = verify_mesh_quality(mesh)
    assert quality.num_shell_elements == 2
    # A triangle is planar by definition, so it contributes no warp.
    assert quality.max_warp == pytest.approx(0.0)
    assert quality.max_aspect_ratio > 1.0
    assert mesh.num_elements == 2
    assert mesh.corners_of(1) == (1, 2, 3)


def test_asking_for_corners_of_a_missing_element_says_so() -> None:
    with pytest.raises(MeshError, match="no shell element 9"):
        Mesh().corners_of(9)


def test_the_advisory_limits_are_the_documented_ones() -> None:
    assert ASPECT_RATIO_LIMIT == pytest.approx(5.0)
    assert WARP_LIMIT == pytest.approx(0.05)


def _round_trip(mesh: Mesh) -> Mesh:
    # Through real JSON text, not just a dict, so a numpy value that json cannot
    # encode fails here rather than at the first save.
    return mesh_from_dict(json.loads(json.dumps(mesh_to_dict(mesh))))


def test_a_stiffened_panel_survives_the_round_trip_intact() -> None:
    mesh = stiffened_panel_mesh(
        StiffenedPanel(
            length=4.0, width=3.0, plate_thickness=0.012, stiffener_spacing=1.5,
            stiffener_height=0.30, stiffener_web_thickness=0.010,
            stiffener_flange_width=0.150, stiffener_flange_thickness=0.015, num_stiffeners=1,
        ),
        PanelMeshConfig(shell_num_divisions_x=3, shell_num_divisions_y=3, beam_num_divisions=2),
    )
    back = _round_trip(mesh)

    assert back.order == mesh.order
    assert sorted(back.nodes) == sorted(mesh.nodes)
    assert all(np.array_equal(back.nodes[k], mesh.nodes[k]) for k in mesh.nodes)
    assert back.quads == mesh.quads
    assert back.beams == mesh.beams
    assert back.couplings == mesh.couplings
    # The association is part of the round trip: without it every restraint named
    # against geometry would have nothing to resolve against.
    assert back.nodes_of_edge == mesh.nodes_of_edge
    assert back.offset_nodes_of_edge == mesh.offset_nodes_of_edge
    assert back.elements_of_face == mesh.elements_of_face
    assert back.elements_of_edge == mesh.elements_of_edge
    assert back.thickness_of_face == mesh.thickness_of_face


def test_a_mapped_mesh_keeps_its_grid_and_vertices() -> None:
    model = GeometryModel()
    model.add_face(model.add_polyline(model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)]), close=True))
    mesh = generate_mesh(model, target_size=0.5, order="quadratic")
    back = _round_trip(mesh)

    assert back.node_of_vertex == mesh.node_of_vertex
    assert set(back.grid_of_face) == set(mesh.grid_of_face)
    for face_id, grid in mesh.grid_of_face.items():
        assert np.array_equal(back.grid_of_face[face_id], grid)
    assert back.is_quadratic


def test_the_seeding_is_not_claimed_after_a_round_trip() -> None:
    model = GeometryModel()
    model.add_face(model.add_polyline(model.add_points([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]), close=True))
    mesh = generate_mesh(model, target_size=0.5)

    assert mesh.seeding is not None
    # A reloaded mesh admits it has no seeding rather than inventing one it could
    # not reproduce.
    assert _round_trip(mesh).seeding is None


def test_saving_refuses_to_overwrite_unless_asked(tmp_path) -> None:
    mesh = simple_panel_mesh(1.0, 1.0, 0.01, 2, 2)
    path = tmp_path / "mesh.json"

    save_mesh(path, mesh)
    assert load_mesh(path).quads == mesh.quads
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_mesh(path, mesh)
    save_mesh(path, mesh, overwrite=True)


def test_a_foreign_or_broken_document_is_refused(tmp_path) -> None:
    with pytest.raises(MeshError, match="not an anymesher.mesh document"):
        mesh_from_dict({"format": "something-else", "version": 1})
    with pytest.raises(MeshError, match="unsupported anymesher.mesh version"):
        mesh_from_dict({"format": "anymesher.mesh", "version": 99})

    document = mesh_to_dict(simple_panel_mesh(1.0, 1.0, 0.01, 1, 1))
    document["quads"]["1"] = [1, 2, 3, 999]
    with pytest.raises(MeshError, match="undefined node"):
        mesh_from_dict(document)
