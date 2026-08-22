from __future__ import annotations

from anygeometry import EntityRef, GeometryModel
from anygeometry.serialization import to_dict

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.refinement import Refinement


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
