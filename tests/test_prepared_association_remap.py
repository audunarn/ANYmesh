from __future__ import annotations

import numpy as np
import pytest

from anygeometry import EntityRef, GeometryModel, split_face_at
from anygeometry.serialization import to_dict
from anymesher import Mesh, MeshError
from anymesher.mapped import generate_mesh
from anymesher.prepared import remap_prepared_mesh_associations


def _partitioned_rectangle():
    source = GeometryModel()
    face = source.add_plate(
        source.add_points(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
             (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    sheet = source.add_sheet((face,))
    source_snapshot = to_dict(source)
    working = source.clone()
    split_face_at(working, face, 0, 0.5)
    face_map = {
        face: tuple(item.id for item in working.resolve_ref(EntityRef("face", face)))
    }
    edge_map = {
        edge_id: tuple(
            item.id for item in working.resolve_ref(EntityRef("edge", edge_id))
        )
        for edge_id in source.edges
    }
    return source, working, face, sheet, source_snapshot, face_map, edge_map


def test_partition_mesh_remaps_to_source_without_losing_block_grids() -> None:
    source, working, face, sheet, snapshot, face_map, edge_map = (
        _partitioned_rectangle()
    )
    working_faces = face_map[face]
    mesh = generate_mesh(working, target_size=0.25)
    working_elements = {
        element
        for working_face in working_faces
        for element in mesh.elements_of_face[working_face]
    }

    result = remap_prepared_mesh_associations(
        mesh,
        source,
        working,
        source_to_working_faces=face_map,
        source_to_working_edges=edge_map,
    )

    assert to_dict(source) == snapshot
    assert result.geometry_model_id == source.model_id
    assert result.geometry_revision == source.revision
    assert result.elements_of_face == {face: sorted(working_elements)}
    assert face not in result.grid_of_face
    assert len(result.block_grids_of_face[face]) == 2
    assert result.nodes_on(EntityRef("face", face))
    assert result.elements_of_sheet[sheet] == sorted(working_elements)
    assert set(result.nodes_of_edge) == set(source.edges)
    assert set(result.elements_of_face).isdisjoint(
        set(working_faces).difference({face})
    )
    assert result.seeding is not None
    for edge_id, nodes in result.nodes_of_edge.items():
        assert result.seeding[edge_id] == len(nodes) - 1
        parameters = [
            source.closest_edge_point(edge_id, result.nodes[node])[1]
            for node in nodes
        ]
        assert parameters == sorted(parameters)


def test_remap_rejects_missing_or_multiply_claimed_descendants_atomically() -> None:
    source, working, face, _sheet, _snapshot, face_map, edge_map = (
        _partitioned_rectangle()
    )
    mesh = generate_mesh(working, target_size=0.25)
    original_identity = mesh.geometry_model_id
    bad_map = dict(face_map)
    bad_map[face] = (*face_map[face], 999_999)

    with pytest.raises(MeshError, match="missing working face"):
        remap_prepared_mesh_associations(
            mesh,
            source,
            working,
            source_to_working_faces=bad_map,
            source_to_working_edges=edge_map,
        )

    assert mesh.geometry_model_id == original_identity
    assert set(mesh.elements_of_face) == set(face_map[face])


def test_remap_allows_exact_coincident_source_edges_to_share_a_descendant() -> None:
    source = GeometryModel()
    first_start, first_end, second_start, second_end = source.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )
    first = source.add_line(first_start, first_end)
    second = source.add_line(second_start, second_end)
    working = source.clone()
    working.remove_edge(second)
    mesh = Mesh(
        geometry_model_id=working.model_id,
        geometry_revision=working.revision,
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
        },
        nodes_of_edge={first: [1, 2]},
    )

    result = remap_prepared_mesh_associations(
        mesh,
        source,
        working,
        source_to_working_faces={},
        source_to_working_edges={first: (first,), second: (first,)},
    )

    assert result.nodes_of_edge == {first: [1, 2], second: [1, 2]}
