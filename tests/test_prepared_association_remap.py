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


def _two_lines(second_end_x: float):
    source = GeometryModel()
    a, b, c, d = source.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (second_end_x, 0.0, 0.0),
        )
    )
    first = source.add_line(a, b)
    second = source.add_line(c, d)
    working = source.clone()
    working.remove_edge(second)
    mesh = Mesh(
        geometry_model_id=working.model_id,
        geometry_revision=working.revision,
        nodes={1: np.asarray((0.0, 0.0, 0.0)), 2: np.asarray((1.0, 0.0, 0.0))},
        nodes_of_edge={first: [1, 2]},
    )
    return source, working, mesh, first, second


def test_remap_allows_exact_coincident_source_edges_to_share_a_descendant() -> None:
    source, working, mesh, first, second = _two_lines(1.0)

    result = remap_prepared_mesh_associations(
        mesh,
        source,
        working,
        source_to_working_faces={},
        source_to_working_edges={first: (first,), second: (first,)},
    )

    assert result.nodes_of_edge == {first: [1, 2], second: [1, 2]}


def test_remap_still_refuses_shared_descendant_for_non_coincident_edges() -> None:
    source, working, mesh, first, second = _two_lines(2.0)

    with pytest.raises(MeshError, match="claimed by source edges"):
        remap_prepared_mesh_associations(
            mesh,
            source,
            working,
            source_to_working_faces={},
            source_to_working_edges={first: (first,), second: (first,)},
        )
    assert mesh.nodes_of_edge == {first: [1, 2]}


def _shared_edge_ids():
    source, working, _mesh, first, second = _two_lines(1.0)
    return source, working, first, second, {first: (first,), second: (first,)}


@pytest.mark.parametrize("pair, ok", [((2, 4), False), ((3, 3), True)])
def test_shared_descendant_division_requests_must_agree(pair, ok) -> None:
    from anymesher.hybrid import _remap_edge_divisions

    source, working, first, second, table = _shared_edge_ids()
    overrides = {first: pair[0], second: pair[1]}
    if ok:
        assert _remap_edge_divisions(source, working, table, overrides) == {first: 3}
    else:
        with pytest.raises(MeshError, match="conflicting"):
            _remap_edge_divisions(source, working, table, overrides)


@pytest.mark.parametrize("sizes, ok", [((0.5, 0.25), False), ((0.5, 0.5), True)])
def test_shared_descendant_size_overrides_must_agree(sizes, ok) -> None:
    from anymesher.hybrid import _remap_entity_overrides
    from anymesher.refinement import size_override

    _source, _working, first, second, table = _shared_edge_ids()
    items = (
        size_override(EntityRef("edge", first), sizes[0], name="a"),
        size_override(EntityRef("edge", second), sizes[1], name="b"),
    )
    if ok:
        result = _remap_entity_overrides(items, {}, table)
        assert {item.size for item in result} == {0.5}
    else:
        with pytest.raises(MeshError, match="conflicting"):
            _remap_entity_overrides(items, {}, table)


def test_same_source_edge_named_twice_keeps_finer_request() -> None:
    from anymesher.hybrid import _remap_entity_overrides
    from anymesher.refinement import size_override

    _source, _working, first, _second, table = _shared_edge_ids()
    items = (
        size_override(EntityRef("edge", first), 0.5, name="a"),
        size_override(EntityRef("edge", first), 0.25, name="b"),
    )
    assert [item.size for item in _remap_entity_overrides(items, {}, table)] == [0.5, 0.25]


# ---------------------------------------------------------------- round 2


def _items(spec):
    from anymesher.refinement import size_override

    return [
        size_override(EntityRef("edge", edge), size, name=f"{edge}-{size}")
        for edge, size in spec
    ]


def _effective_edge_size(working, items, edge):
    from anymesher.refinement import SizeField

    return SizeField(working, 1.0, items).edge_size(edge)


@pytest.mark.parametrize(
    "per_source, expect",
    [
        # A names 0.5 then 0.25 (finer wins -> 0.25); B names 0.5: disagree.
        (((1, 0.5), (1, 0.25), (2, 0.5)), None),
        # A resolves to 0.25 and B names 0.25: agree.
        (((1, 0.5), (1, 0.25), (2, 0.25)), 0.25),
        # Plain agreement and plain conflict.
        (((1, 0.5), (2, 0.5)), 0.5),
        (((1, 0.5), (2, 0.25)), None),
    ],
)
def test_shared_descendant_size_resolution_is_order_independent(per_source, expect) -> None:
    from itertools import permutations

    from anymesher.hybrid import _remap_entity_overrides

    source, working, first, second, _ = _shared_edge_ids()
    assert (first, second) == (1, 2)
    table = {first: (first,), second: (first,)}
    for order in set(permutations(per_source)):
        items = _items(order)
        if expect is None:
            with pytest.raises(MeshError, match="conflicting"):
                _remap_entity_overrides(items, {}, table)
            continue
        result = _remap_entity_overrides(items, {}, table)
        # every record survives, rebound to the shared descendant
        assert [(r.name, r.size) for r in result] == [(r.name, r.size) for r in items]
        assert all(r.ref == EntityRef("edge", first) for r in result)
        assert _effective_edge_size(working, result, first) == expect


def test_unshared_override_sequence_is_unchanged() -> None:
    from anymesher.hybrid import _remap_entity_overrides

    items = _items(((1, 0.5), (1, 0.25), (2, 0.5)))
    result = _remap_entity_overrides(items, {}, {1: (1,), 2: (2,)})
    assert result == tuple(items)


def _shared_vertex_case(*, reverse: bool, bindings):
    source = GeometryModel()
    a, b, c, d = source.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0) if reverse else (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0) if reverse else (1.0, 0.0, 0.0),
        )
    )
    first = source.add_line(a, b)
    second = source.add_line(c, d)
    working = source.clone()
    working.remove_edge(second)
    ids = {"a": a, "b": b, "c": c, "d": d}
    mesh = Mesh(
        geometry_model_id=working.model_id,
        geometry_revision=working.revision,
        nodes={1: np.asarray((0.0, 0.0, 0.0)), 2: np.asarray((1.0, 0.0, 0.0))},
        nodes_of_edge={first: [1, 2]},
        node_of_vertex={ids[k]: v for k, v in bindings.items()},
    )
    return source, working, mesh, first, second, ids


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_edge_endpoint_vertices_of_every_source_resolve(reverse) -> None:
    source, working, mesh, first, second, v = _shared_vertex_case(
        reverse=reverse, bindings={"a": 1, "b": 2}
    )
    result = remap_prepared_mesh_associations(
        mesh, source, working,
        source_to_working_faces={},
        source_to_working_edges={first: (first,), second: (first,)},
    )
    assert result.node_of_vertex[v["a"]] == 1 and result.node_of_vertex[v["b"]] == 2
    # second source: start vertex sits at node 2 when the edge is reversed
    assert result.node_of_vertex[v["c"]] == (2 if reverse else 1)
    assert result.node_of_vertex[v["d"]] == (1 if reverse else 2)


def test_shared_edge_vertex_binding_contradiction_is_refused() -> None:
    source, working, mesh, first, second, v = _shared_vertex_case(
        reverse=False, bindings={"a": 1, "b": 2, "c": 2}
    )
    before = dict(mesh.node_of_vertex)
    with pytest.raises(MeshError, match="vertex"):
        remap_prepared_mesh_associations(
            mesh, source, working,
            source_to_working_faces={},
            source_to_working_edges={first: (first,), second: (first,)},
        )
    assert mesh.node_of_vertex == before


def test_unshared_edges_and_unrelated_vertices_gain_no_vertex_binding() -> None:
    source, working, mesh, first, second, v = _shared_vertex_case(
        reverse=False, bindings={"a": 1, "b": 2}
    )
    extra = source.add_points(((5.0, 5.0, 5.0),))[0]
    working = source.clone()
    working.remove_edge(second)
    result = remap_prepared_mesh_associations(
        mesh, source, working,
        source_to_working_faces={},
        source_to_working_edges={first: (first,), second: (first,)},
    )
    assert extra not in result.node_of_vertex
