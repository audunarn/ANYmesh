"""The public mapped generator consumes a source-named, ordered interface."""
import numpy as np
import pytest

from anygeometry import GeometryModel
from anymesher import mapped
from anymesher.errors import MeshError
from anymesher.regional_mapped import remesh_mapped_region
from anymesher.regional_update import RegionalQualityLimits
from anymesher.seeding import solve_seeding
from anymesher.serialize import mesh_to_dict


def request(order):
    model = GeometryModel()
    a, b, c, d, e, f = model.add_points(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]
    )
    left_edges = model.add_polyline([a, b, c, d], close=True)
    shared = left_edges[1]
    left = model.add_face(left_edges)
    right_edges = [*model.add_polyline([b, e, f, c]), shared]
    right = model.add_face(right_edges)
    previous = mapped.generate_mesh(model, target_size=0.5, order=order)
    seed = solve_seeding(model, target_size=0.5, overrides={right_edges[0]: 4})
    args = dict(affected_faces=(right,), interface_stations={shared: tuple(previous.nodes_of_edge[shared])},
                interface_vertices={v: previous.node_of_vertex[v] for v in (b, c)},
                seeding=seed, target_size=0.5, node_high_water=1000, element_high_water=2000,
                quality=RegionalQualityLimits(max_aspect_ratio=4., min_scaled_jacobian=0.2))
    return model, previous, left, shared, args


@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_real_owner_regeneration_preserves_all_stations_and_exterior(order):
    geometry, old, left, edge, args = request(order)
    original = mesh_to_dict(old)
    original_counts = dict(args["seeding"].divisions)
    result = remesh_mapped_region(old, geometry, **args)
    assert mesh_to_dict(old) == original
    assert args["seeding"].divisions == original_counts
    assert result.stations_validated
    assert set(result.interface_node_ids) == set(old.nodes_of_edge[edge])
    assert tuple(result.mesh.nodes_of_edge[edge]) == tuple(old.nodes_of_edge[edge])
    assert tuple(result.retained_element_ids) == tuple(sorted(old.elements_of_face[left]))
    assert all(result.mesh.quads[e] == old.quads[e] for e in result.retained_element_ids)
    assert all(np.array_equal(result.mesh.nodes[n], old.nodes[n]) for n in result.retained_node_ids)
    assert len(result.new_element_ids) == 8 and len(result.retained_element_ids) == 4
    assert min(result.new_element_ids) > 2000 and min(result.new_node_ids) > 1000
    assert result.seeding_changed_edges
    assert len(result.mesh.nodes_of_edge[edge]) == (3 if order == "linear" else 5)


@pytest.mark.parametrize("corruption", ["reversed", "missing_mid", "foreign_vertex", "missing_edge"])
def test_refuses_incomplete_or_foreign_prior_proof_without_mutation(corruption):
    geometry, old, _, edge, args = request("quadratic")
    original = mesh_to_dict(old)
    seq = args["interface_stations"][edge]
    if corruption == "reversed":
        args["interface_stations"][edge] = seq[::-1]
    elif corruption == "missing_mid":
        args["interface_stations"][edge] = seq[::2]
    elif corruption == "foreign_vertex":
        v = next(iter(args["interface_vertices"]))
        args["interface_vertices"][v] = 11111
    else:
        args["interface_stations"].clear()
    with pytest.raises(MeshError, match="frozen"):
        remesh_mapped_region(old, geometry, **args)
    assert mesh_to_dict(old) == original
