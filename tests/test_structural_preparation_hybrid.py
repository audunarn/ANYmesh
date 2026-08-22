from __future__ import annotations

from anygeometry import EntityRef, GeometryModel
from anygeometry.serialization import to_dict
import pytest

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher import MeshError
from anymesher.refinement import Refinement, SizeField
from anymesher.seeding import solve_seeding


def _crossing_faces() -> tuple[GeometryModel, int, int]:
    geometry = GeometryModel()
    horizontal = geometry.add_plate(
        geometry.add_points(
            ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        )
    )
    vertical = geometry.add_plate(
        geometry.add_points(
            ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1))
        )
    )
    return geometry, horizontal, vertical


def test_hybrid_automatically_prepares_crossing_plates_on_clone() -> None:
    geometry, first, second = _crossing_faces()
    before = to_dict(geometry)

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        strategy="auto",
        native_backend="python",
    )

    assert to_dict(geometry) == before
    assert result.structural_preparation is not None
    assert result.structural_preparation.face_connections == 1
    assert result.mesh.automatic_intersections == 1
    assert result.mesh.geometry_model_id == geometry.model_id
    assert result.mesh.geometry_revision == geometry.revision
    first_nodes = set(result.mesh.nodes_on(EntityRef("face", first)))
    second_nodes = set(result.mesh.nodes_on(EntityRef("face", second)))
    assert len(first_nodes.intersection(second_nodes)) >= 3
    assert result.mesh.structural_preparation["structural_closure"][
        "preparation_hash"
    ].startswith("sha256:")


def test_hybrid_automatically_connects_crossing_beams() -> None:
    geometry = GeometryModel()
    first = geometry.add_line(
        *geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    )
    second = geometry.add_line(
        *geometry.add_points(((0, -1, 0), (0, 1, 0)))
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        beam_edges=(first, second),
    )

    assert result.structural_preparation is not None
    assert result.structural_preparation.member_connections == 1
    first_nodes = set(result.mesh.nodes_on(EntityRef("edge", first)))
    second_nodes = set(result.mesh.nodes_on(EntityRef("edge", second)))
    assert len(first_nodes.intersection(second_nodes)) == 1
    assert result.mesh.elements_of_edge[first]
    assert result.mesh.elements_of_edge[second]


def test_hybrid_automatically_declares_beam_to_shell_attachment() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        )
    )
    edge = geometry.add_line(
        *geometry.add_points(((0, 0, -1), (0, 0, 1)))
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        face_ids=(face,),
        beam_edges=(edge,),
        native_backend="python",
    )

    assert result.structural_preparation is not None
    assert result.structural_preparation.member_sheet_connections == 1
    assert result.mesh.couplings
    assert not result.connectivity.issues


def test_clone_only_mode_preserves_vertex_bound_refinement() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    )
    edge = geometry.faces[face].loop[0].edge
    vertex = geometry.edges[edge].start
    before = to_dict(geometry)

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        face_ids=(face,),
        refinements=(
            Refinement(
                size=0.25,
                radius=0.5,
                ref=EntityRef("vertex", vertex),
            ),
        ),
        structural_preparation=False,
    )

    assert result.mesh.elements_of_face[face]
    assert result.structural_preparation is not None
    assert result.structural_preparation.applications == 0
    assert to_dict(geometry) == before


def test_default_coplanar_stiffener_connects_and_retains_eccentricity() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    )
    beam = geometry.add_line(
        *geometry.add_points(((0, 0.5, 0), (2, 0.5, 0)))
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        face_ids=(face,),
        beam_edges=(beam,),
        beam_offsets={beam: 0.1},
    )

    assert result.structural_preparation is not None
    assert result.structural_preparation.member_sheet_connections == 1
    assert result.mesh.offset_nodes_of_edge[beam]
    assert len(result.mesh.offset_nodes_of_edge[beam]) == len(
        result.mesh.nodes_of_edge[beam]
    )
    assert result.mesh.couplings
    assert not result.connectivity.issues


def test_sheet_attachment_uses_recorded_face_not_same_numbered_face() -> None:
    geometry = GeometryModel()
    target = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    )
    unrelated = geometry.add_plate(
        geometry.add_points(((5, 0, 0), (6, 0, 0), (6, 1, 0), (5, 1, 0)))
    )
    geometry.add_sheet((unrelated,))
    target_sheet = geometry.add_sheet((target,))
    assert target_sheet != target
    beam = geometry.add_line(
        *geometry.add_points(((0, 0.5, 0), (2, 0.5, 0)))
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        face_ids=(target,),
        beam_edges=(beam,),
    )

    assert result.structural_preparation is not None
    assert result.structural_preparation.member_sheet_connections == 1
    assert result.mesh.couplings
    assert not result.connectivity.issues


def test_partly_outside_eccentric_stiffener_fails_closed() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    )
    beam = geometry.add_line(
        *geometry.add_points(((-0.5, 0.5, 0), (1.0, 0.5, 0)))
    )
    before = to_dict(geometry)

    with pytest.raises(MeshError, match="outside every exactly attached plate range"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.25,
            face_ids=(face,),
            beam_edges=(beam,),
            beam_offsets={beam: 0.1},
        )

    assert to_dict(geometry) == before


def test_split_edge_override_keeps_the_explicit_source_total() -> None:
    geometry = GeometryModel()
    horizontal = geometry.add_line(
        *geometry.add_points(((0, 0, 0), (1, 0, 0)))
    )
    first = geometry.add_line(
        *geometry.add_points(((0.9, -1, 0), (0.9, 1, 0)))
    )
    second = geometry.add_line(
        *geometry.add_points(((0.95, -1, 0), (0.95, 1, 0)))
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        face_ids=(),
        beam_edges=(horizontal, first, second),
        overrides={horizontal: 3},
    )

    assert result.structural_preparation is not None
    assert len(result.structural_preparation.source_to_working_edges[horizontal]) == 3
    assert result.mesh.seeding is not None
    assert result.mesh.seeding.divisions[horizontal] == 3


def test_split_edge_override_fails_when_each_descendant_cannot_get_one() -> None:
    geometry = GeometryModel()
    horizontal = geometry.add_line(
        *geometry.add_points(((0, 0, 0), (1, 0, 0)))
    )
    first = geometry.add_line(
        *geometry.add_points(((0.4, -1, 0), (0.4, 1, 0)))
    )
    second = geometry.add_line(
        *geometry.add_points(((0.8, -1, 0), (0.8, 1, 0)))
    )

    with pytest.raises(MeshError, match="created 3 non-empty descendants"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.25,
            face_ids=(),
            beam_edges=(horizontal, first, second),
            overrides={horizontal: 2},
        )


def test_split_embedded_stiffener_retains_offset_on_every_descendant() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    )
    horizontal = geometry.add_line(
        *geometry.add_points(((0, 1, 0), (2, 1, 0)))
    )
    vertical = geometry.add_line(
        *geometry.add_points(((1, 0, 0), (1, 2, 0)))
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        face_ids=(face,),
        beam_edges=(horizontal, vertical),
        beam_offsets={horizontal: 0.1, vertical: 0.1},
    )

    assert result.structural_preparation is not None
    assert len(result.structural_preparation.source_to_working_edges[horizontal]) == 2
    assert len(result.structural_preparation.source_to_working_edges[vertical]) == 2
    assert len(result.mesh.offset_nodes_of_edge[horizontal]) == len(
        result.mesh.nodes_of_edge[horizontal]
    )
    assert len(result.mesh.offset_nodes_of_edge[vertical]) == len(
        result.mesh.nodes_of_edge[vertical]
    )
    assert not result.connectivity.issues


def test_identity_preparation_preserves_seeding_classes_and_size_field() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    )
    edges = tuple(item.edge for item in geometry.faces[face].loop)
    size_field = SizeField(geometry, 0.25)
    expected = solve_seeding(
        geometry,
        size_field=size_field,
        edge_ids=edges,
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        face_ids=(face,),
    )

    assert result.mesh.seeding is not None
    assert result.mesh.seeding.divisions == expected.divisions
    assert result.mesh.seeding.classes == expected.classes
    assert result.mesh.seeding.size_field is not None
    assert result.mesh.seeding.size_field.target_size == 0.25
