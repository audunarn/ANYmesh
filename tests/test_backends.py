"""Backend dispatch, and the gmsh backend.

The gmsh tests are skipped when gmsh is not installed, since it is an optional
extra.  What they check is not that gmsh works -- that is gmsh's problem -- but
that what comes back is a mesh this package's own machinery can use, and that the
two guarantees the mapped mesher makes and gmsh does not are stated rather than
quietly broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from anymesher import (
    DEFAULT_BACKEND,
    EntityRef,
    GeometryModel,
    MeshError,
    available_backends,
    generate_mesh,
    resolve_backend,
    verify_mesh_quality,
)

gmsh = pytest.importorskip("gmsh", reason="the gmsh backend needs ANYmesher[gmsh]")


def _rectangle(length: float = 2.0, width: float = 1.0) -> GeometryModel:
    model = GeometryModel()
    model.add_face(
        model.add_polyline(
            model.add_points([(0, 0, 0), (length, 0, 0), (length, width, 0), (0, width, 0)]),
            close=True,
        )
    )
    return model


def test_the_default_backend_is_the_mapped_mesher() -> None:
    assert DEFAULT_BACKEND == "mapped"
    assert "mapped" in available_backends()
    assert "gmsh" in available_backends()

    from anymesher.mapped import generate_mesh as mapped_generate

    assert resolve_backend("mapped") is mapped_generate
    assert resolve_backend("MAPPED") is mapped_generate


def test_dispatching_with_the_default_matches_calling_the_mesher_directly() -> None:
    from anymesher.mapped import generate_mesh as mapped_generate

    dispatched = generate_mesh(_rectangle(), target_size=0.25)
    direct = mapped_generate(_rectangle(), target_size=0.25)

    assert dispatched.quads == direct.quads
    assert all(np.array_equal(dispatched.nodes[k], direct.nodes[k]) for k in direct.nodes)


def test_an_unknown_backend_names_the_ones_that_exist() -> None:
    with pytest.raises(MeshError, match="unknown mesh backend 'quadratic'"):
        generate_mesh(_rectangle(), backend="quadratic", target_size=0.25)
    with pytest.raises(MeshError, match="gmsh, mapped"):
        resolve_backend("nope")


@pytest.mark.parametrize("order,nodes_per_quad", [("linear", 4), ("quadratic", 8)])
def test_gmsh_returns_a_usable_mesh(order: str, nodes_per_quad: int) -> None:
    mesh = generate_mesh(_rectangle(), backend="gmsh", target_size=0.25, order=order)

    assert mesh.order == order
    assert mesh.quads
    assert all(len(nodes) == nodes_per_quad for nodes in mesh.quads.values())
    # Every element node exists, or nothing downstream can look one up.
    assert all(node in mesh.nodes for nodes in mesh.shells.values() for node in nodes)
    # Serendipity, not Lagrange: no centre node on a quadratic quad.
    assert nodes_per_quad != 9

    quality = verify_mesh_quality(mesh)
    assert quality.num_shell_elements == len(mesh.quads) + len(mesh.tris)
    assert np.isfinite(quality.max_aspect_ratio)


def test_gmsh_fills_the_association_so_results_stay_addressable() -> None:
    model = _rectangle()
    mesh = generate_mesh(model, backend="gmsh", target_size=0.25)

    assert mesh.elements_of_face[1]
    assert mesh.nodes_on(EntityRef("face", 1))
    for edge_id in model.edges:
        nodes = mesh.nodes_on(EntityRef("edge", edge_id))
        assert len(nodes) >= 2
        # Ordered along the edge: consecutive nodes must be monotonic in arc
        # length, or a load distributed along a line lands out of order.
        positions = np.array([mesh.nodes[node] for node in nodes])
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert np.all(steps > 0.0)
        span = float(np.linalg.norm(positions[-1] - positions[0]))
        assert float(np.sum(steps)) == pytest.approx(span, rel=1.0e-6)
    for vertex_id in model.vertices:
        assert vertex_id in mesh.node_of_vertex


def test_gmsh_has_no_structured_grid_and_says_so_by_leaving_it_empty() -> None:
    mesh = generate_mesh(_rectangle(), backend="gmsh", target_size=0.3)

    # The mapped mesher's (i, j) index does not exist for an unstructured mesh.
    # It is left empty rather than filled with something plausible.
    assert mesh.grid_of_face == {}
    # nodes_on still works, through the element groups instead of the grid.
    assert mesh.nodes_on(EntityRef("face", 1))


def test_gmsh_keeps_triangles_rather_than_dropping_them() -> None:
    mesh = generate_mesh(_rectangle(), backend="gmsh", target_size=0.3, recombine=False)

    assert mesh.tris
    assert not mesh.quads
    assert mesh.num_elements == len(mesh.tris)
    assert all(len(nodes) == 3 for nodes in mesh.tris.values())
    assert all(node in mesh.nodes for nodes in mesh.tris.values() for node in nodes)


def test_gmsh_puts_arc_nodes_exactly_on_the_arc() -> None:
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    via = model.add_point(1.0, 1.2, 0.0)
    edges = [
        model.add_line(points[0], points[1]),
        model.add_line(points[1], points[2]),
        model.add_arc(points[2], via, points[3]),
        model.add_line(points[3], points[0]),
    ]
    model.add_face(edges)

    mesh = generate_mesh(model, backend="gmsh", target_size=0.25)
    frame = model.arc_frame(edges[2])
    radii = np.linalg.norm(
        np.array([mesh.nodes[node] for node in mesh.nodes_on(EntityRef("edge", edges[2]))])
        - frame.center,
        axis=1,
    )
    assert radii == pytest.approx(frame.radius)


def test_gmsh_refuses_a_non_planar_face_and_points_at_the_mapped_mesher() -> None:
    model = GeometryModel()
    radius = 2.0
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    model.extrude([model.add_arc(start, via, end)], (0.0, 0.0, 3.0))

    with pytest.raises(MeshError, match="not planar"):
        generate_mesh(model, backend="gmsh", target_size=0.4)
    with pytest.raises(MeshError, match="backend='mapped'"):
        generate_mesh(model, backend="gmsh", target_size=0.4)

    # The mapped mesher meshes it exactly, which is why refusing is reasonable.
    assert generate_mesh(model, target_size=0.4).quads


def test_gmsh_validates_its_own_inputs() -> None:
    model = _rectangle()

    with pytest.raises(MeshError, match="target_size must be positive"):
        generate_mesh(model, backend="gmsh", target_size=0.0)
    with pytest.raises(MeshError, match="unknown element order"):
        generate_mesh(model, backend="gmsh", target_size=0.3, order="cubic")
    with pytest.raises(MeshError, match="no face 99"):
        generate_mesh(model, backend="gmsh", target_size=0.3, face_ids=[99])
    with pytest.raises(MeshError, match="no edge 99"):
        generate_mesh(model, backend="gmsh", target_size=0.3, beam_edges=[99])


def test_gmsh_is_finalized_even_when_meshing_fails() -> None:
    model = GeometryModel()
    radius = 2.0
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    model.extrude([model.add_arc(start, via, end)], (0.0, 0.0, 3.0))

    with pytest.raises(MeshError):
        generate_mesh(model, backend="gmsh", target_size=0.4)

    # gmsh is process-global state.  A backend that left it initialized would
    # break the next unrelated call, so initializing again must succeed.
    gmsh.initialize()
    gmsh.finalize()
    assert generate_mesh(_rectangle(), backend="gmsh", target_size=0.3).quads


def test_gmsh_meshes_beam_edges_along_their_own_direction() -> None:
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)

    mesh = generate_mesh(model, backend="gmsh", target_size=0.25, beam_edges=[edges[0]])
    assert mesh.beams
    assert mesh.elements_of_edge[edges[0]]
    for connectivity in mesh.beams.values():
        assert len(connectivity) == 2
        assert all(node in mesh.nodes for node in connectivity)
