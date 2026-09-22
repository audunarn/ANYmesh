from __future__ import annotations

import numpy as np
import pytest

from anygeometry import GeometryModel, punch_hole
from anymesher.quad.boundary import BoundaryStationRegistry
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.seed import build_planar_quad_seed
from anymesher.triangulation import orient2d
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions


def _signature(geometry) -> tuple:
    return (
        str(geometry.model_id),
        int(geometry.revision),
        tuple(
            (i, tuple(map(float, geometry.vertex_position(i))))
            for i in sorted(geometry.vertices)
        ),
        tuple((i, geometry.edges[i].start, geometry.edges[i].end) for i in sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
    )


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x = float(point[0]); y = float(point[1])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            if x < xj + (y - yj) * (xi - xj) / (yi - yj):
                inside = not inside
        j = i
    return inside


def _shoelace(chart) -> float:
    n = len(chart)
    return 0.5 * sum(chart[i][0] * chart[(i + 1) % n][1] - chart[(i + 1) % n][0] * chart[i][1] for i in range(n))


def _rectangle_face() -> tuple:
    points = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 6.0, 0.0), (0.0, 6.0, 0.0))
    geometry = GeometryModel()
    vertices = geometry.add_points(points)
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
    return geometry, face


def _p03_geometry() -> tuple:
    geometry, face = _rectangle_face()
    face, hole_edges = punch_hole(geometry, face, (3.2, 2.4, 0.0), 0.9)
    return geometry, face, tuple(hole_edges)


def _keys(registry: BoundaryStationRegistry, edge: int, reverse: bool = False) -> tuple:
    return tuple(s.key for s in registry.chain(edge, reverse))


def _outer_station_keys(registry: BoundaryStationRegistry, domain: PlanarQuadDomain) -> tuple:
    out: list = []
    for edge, forward in domain.edge_uses:
        out.extend(s.key for s in registry.chain(edge, forward)[:-1])
    return tuple(dict.fromkeys(out))


def test_p03_domain_records_hole_and_registry_covers_every_outer_hole_edge() -> None:
    geometry, face, hole_edges = _p03_geometry()
    before = _signature(geometry)

    domain = PlanarQuadDomain.from_geometry(geometry, face)
    assert len(domain.hole_vertex_loops) == 1
    assert len(domain.hole_edge_uses) == 1
    expected_hole = abs(_shoelace(domain.hole_charts[0]))
    assert expected_hole == pytest.approx(2.0 * 0.81, rel=1.0e-9)
    assert domain.hole_areas == (pytest.approx(expected_hole, rel=1.0e-9),)
    assert domain.outer_area == pytest.approx(60.0, rel=1.0e-9)
    assert domain.area == pytest.approx(60.0 - expected_hole, rel=1.0e-9)
    assert domain.area > 0.0

    registry = BoundaryStationRegistry.for_domain(geometry, domain, 0.5)
    outer = [edge for edge, _ in domain.edge_uses]
    hole = [edge for loop in domain.hole_edge_uses for edge, _ in loop]
    assert sorted(hole) == sorted(hole_edges)
    seen = outer + hole
    for edge in seen:
        assert len(registry.chain(edge)) >= 3
    assert len(seen) == len(set(seen)) == 8
    assert len(outer) == 4 and len(hole) == 4

    outer_edge = domain.edge_uses[0][0]
    assert _keys(registry, outer_edge, False) == tuple(reversed(_keys(registry, outer_edge, True)))
    hole_edge = domain.hole_edge_uses[0][0][0]
    assert _keys(registry, hole_edge, False) == tuple(reversed(_keys(registry, hole_edge, True)))

    face_stations = registry.face_stations(domain)
    assert len(face_stations) == len(_outer_station_keys(registry, domain))
    assert {s.key for s in face_stations} <= set(_outer_station_keys(registry, domain))

    hole_polygon = np.asarray(
        [np.asarray(s.position, dtype=float)[:2] for s in registry.chain(hole_edges[0])]
    )
    assert np.all(np.linalg.norm(np.diff(np.vstack([hole_polygon, hole_polygon[:1]]), axis=0), axis=1) > 0)

    assert _signature(geometry) == before


def test_p03_seed_protects_hole_boundary_and_fills_nothing() -> None:
    geometry, face, _ = _p03_geometry()
    before = _signature(geometry)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    registry = BoundaryStationRegistry.for_domain(geometry, domain, 0.5)
    seed = build_planar_quad_seed(geometry, face, 0.5, domain=domain, registry=registry)

    state = seed.state
    assert set(state.cell_kinds.values()) == {"T3"}
    assert len(state.cells) > 0
    for cell in state.cells.values():
        assert orient2d(state.position(cell[0]), state.position(cell[1]), state.position(cell[2])) > 0.0
    initial_t3_area = sum(
        0.5
        * abs(
            (state.position(cell[1])[0] - state.position(cell[0])[0])
            * (state.position(cell[2])[1] - state.position(cell[0])[1])
            - (state.position(cell[2])[0] - state.position(cell[0])[0])
            * (state.position(cell[1])[1] - state.position(cell[0])[1])
        )
        for cell in state.cells.values()
    )
    assert seed.discrete_area == pytest.approx(initial_t3_area, rel=1.0e-10)
    assert seed.discrete_area < domain.outer_area

    boundary = {tuple(sorted(map(int, e))) for e in seed.triangulation.boundary_segments}
    assert state.front == frozenset(boundary)
    assert state.protected_edges == frozenset(boundary)
    assert state.protected_nodes == frozenset(seed.station_to_node.values())

    all_stations: list = []
    for edge, forward in domain.edge_uses:
        all_stations.extend(registry.chain(edge, forward))
    for loop in domain.hole_edge_uses:
        for edge, forward in loop:
            all_stations.extend(registry.chain(edge, forward))
    station_keys = {s.key for s in all_stations}
    assert station_keys == set(seed.station_to_node)
    for station in all_stations:
        assert seed.station_to_node[station.key] in state.protected_nodes

    hole_edges = [edge for loop in domain.hole_edge_uses for edge, _ in loop]
    hole_polygon = np.asarray(list(dict.fromkeys(
        tuple(map(float, np.asarray(s.position, dtype=float)[:2]))
        for edge in hole_edges
        for s in registry.chain(edge)
    )))
    interior_ids = set(state.nodes) - set(state.protected_nodes)
    lattice_points = [np.asarray(state.position(n), dtype=float)[:2] for n in interior_ids]
    centroids = [
        np.mean([np.asarray(state.position(n), dtype=float)[:2] for n in cell], axis=0)
        for cell in state.cells.values()
    ]
    for point in lattice_points + centroids:
        assert not _point_in_polygon(point, hole_polygon)

    assert _signature(geometry) == before


def test_p05_concave_face_exact_area_and_deterministic_seed() -> None:
    points = ((0, 0, 0), (6, 0, 0), (6, 2, 0), (3, 2, 0), (3, 5, 0), (0, 5, 0))
    geometry = GeometryModel()
    vertices = geometry.add_points(points)
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)

    domain = PlanarQuadDomain.from_geometry(geometry, face)
    assert domain.outer_area == pytest.approx(21.0, rel=1.0e-9)
    assert domain.area == pytest.approx(21.0, rel=1.0e-9)
    assert domain.hole_vertex_loops == ()
    assert domain.hole_edge_uses == ()
    assert domain.hole_areas == ()

    seed = build_planar_quad_seed(geometry, face, 0.5)
    assert set(seed.state.cell_kinds.values()) == {"T3"}
    assert len(seed.state.cells) > 0
    for cell in seed.state.cells.values():
        assert orient2d(seed.state.position(cell[0]), seed.state.position(cell[1]), seed.state.position(cell[2])) > 0.0

    again = build_planar_quad_seed(geometry, face, 0.5)
    assert len(seed.state.nodes) == len(again.state.nodes)
    assert len(seed.state.cells) == len(again.state.cells)
    assert seed.state.front == again.state.front
    for key, pos in seed.state.nodes.items():
        assert np.allclose(np.asarray(pos), np.asarray(again.state.position(key)), atol=1.0e-12)


def test_p03b_two_disjoint_holes_recorded_and_seed_deterministic_without_fill() -> None:
    geometry, face = _rectangle_face()
    face, hole_a = punch_hole(geometry, face, (3.0, 2.0, 0.0), 0.8)
    face, hole_b = punch_hole(geometry, face, (7.0, 4.0, 0.0), 0.8)
    before = _signature(geometry)

    domain = PlanarQuadDomain.from_geometry(geometry, face)
    assert len(domain.hole_vertex_loops) == 2
    assert len(domain.hole_edge_uses) == 2
    expected_a = abs(_shoelace(domain.hole_charts[0]))
    expected_b = abs(_shoelace(domain.hole_charts[1]))
    assert expected_a == pytest.approx(2.0 * 0.64, rel=1.0e-9)
    assert expected_b == pytest.approx(2.0 * 0.64, rel=1.0e-9)
    assert domain.hole_areas == (
        pytest.approx(expected_a, rel=1.0e-9),
        pytest.approx(expected_b, rel=1.0e-9),
    )
    assert domain.area == pytest.approx(60.0 - expected_a - expected_b, rel=1.0e-9)
    assert domain.area > 0.0

    registry = BoundaryStationRegistry.for_domain(geometry, domain, 0.5)
    seen = [edge for edge, _ in domain.edge_uses]
    for loop in domain.hole_edge_uses:
        seen.extend(edge for edge, _ in loop)
    assert len(seen) == len(set(seen)) == 12

    seed = build_planar_quad_seed(geometry, face, 0.5, domain=domain, registry=registry)
    boundary = {tuple(sorted(map(int, e))) for e in seed.triangulation.boundary_segments}
    assert seed.state.front == frozenset(boundary)
    assert seed.state.protected_edges == frozenset(boundary)
    assert seed.state.protected_nodes == frozenset(seed.station_to_node.values())
    assert set(seed.state.cell_kinds.values()) == {"T3"}
    assert len(seed.state.cells) > 0

    for loop in domain.hole_edge_uses:
        points = [
            tuple(map(float, np.asarray(s.position, dtype=float)[:2]))
            for edge, forward in loop
            for s in registry.chain(edge, forward)
        ]
        polygon = np.asarray(list(dict.fromkeys(points)))
        centroids = [
            np.mean([np.asarray(seed.state.position(n), dtype=float)[:2] for n in cell], axis=0)
            for cell in seed.state.cells.values()
        ]
        interior_ids = set(seed.state.nodes) - set(seed.state.protected_nodes)
        lattice_points = [np.asarray(seed.state.position(n), dtype=float)[:2] for n in interior_ids]
        for point in lattice_points + centroids:
            assert not _point_in_polygon(point, polygon)

    again = build_planar_quad_seed(geometry, face, 0.5)
    assert len(seed.state.cells) == len(again.state.cells)
    assert len(seed.state.nodes) == len(again.state.nodes)
    assert seed.state.front == again.state.front

    assert _signature(geometry) == before


def _public_result(geometry, face: int, h: float = 0.5):
    return generate_hybrid_mesh_result(
        geometry,
        target_size=h,
        face_ids=(face,),
        quad_options=QuadMeshingOptions(),
    )


def _public_validation(result, face: int) -> dict:
    mesh = result.mesh
    assert result.strategy_by_face[face] == "quad_first"
    assert mesh.hybrid_diagnostics["route"] == "quad-first"
    return mesh.hybrid_diagnostics["validation"]["faces"][face]


def _assert_public_edge_chains(mesh, domain: PlanarQuadDomain) -> None:
    edge_ids = [edge for edge, _ in domain.edge_uses]
    edge_ids.extend(
        edge for loop in domain.hole_edge_uses for edge, _ in loop
    )
    for edge_id in edge_ids:
        chain = mesh.nodes_of_edge.get(edge_id)
        assert chain is not None, f"missing nodes_of_edge for source edge {edge_id}"
        assert len(chain) >= 2
        assert all(node in mesh.nodes for node in chain)


def _assert_no_hole_centroids(mesh, domain: PlanarQuadDomain, face: int) -> None:
    polygons = tuple(np.asarray(chart, dtype=float) for chart in domain.hole_charts)
    for element_id in mesh.elements_of_face[face]:
        corners = mesh.corners_of(element_id)
        xyz = np.mean(
            [np.asarray(mesh.nodes[node], dtype=float) for node in corners], axis=0
        )
        uv = np.asarray(domain.project(xyz), dtype=float)
        for polygon in polygons:
            assert not _point_in_polygon(uv, polygon)


def _p03b_public_geometry() -> tuple:
    geometry, face = _rectangle_face()
    face, _ = punch_hole(geometry, face, (3.0, 2.0, 0.0), 0.8)
    face, _ = punch_hole(geometry, face, (7.0, 4.0, 0.0), 0.8)
    return geometry, face


def test_p03_public_hole_quad_first() -> None:
    geometry, face, _ = _p03_geometry()
    before = _signature(geometry)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    result = _public_result(geometry, face, 0.5)
    validation = _public_validation(result, face)
    assert _signature(geometry) == before
    assert validation["area_ratio"] == pytest.approx(1.0, rel=1.0e-10, abs=1.0e-10)
    assert validation["q4_count_fraction"] >= 0.75
    assert validation["q4_area_fraction"] >= 0.75
    _assert_public_edge_chains(result.mesh, domain)
    _assert_no_hole_centroids(result.mesh, domain, face)


def test_p05_public_concave_quad_first() -> None:
    points = (
        (0.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
        (6.0, 2.0, 0.0),
        (3.0, 2.0, 0.0),
        (3.0, 5.0, 0.0),
        (0.0, 5.0, 0.0),
    )
    geometry = GeometryModel()
    vertices = geometry.add_points(points)
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
    before = _signature(geometry)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    result = _public_result(geometry, face, 0.5)
    validation = _public_validation(result, face)
    assert _signature(geometry) == before
    assert validation["area_ratio"] == pytest.approx(1.0, rel=1.0e-10, abs=1.0e-10)
    assert validation["q4_count_fraction"] >= 0.75
    assert validation["q4_area_fraction"] >= 0.75
    _assert_public_edge_chains(result.mesh, domain)


def test_p03b_public_two_holes_deterministic() -> None:
    first_geometry, first_face = _p03b_public_geometry()
    second_geometry, second_face = _p03b_public_geometry()
    first_before = _signature(first_geometry)
    second_before = _signature(second_geometry)
    first_domain = PlanarQuadDomain.from_geometry(first_geometry, first_face)
    second_domain = PlanarQuadDomain.from_geometry(second_geometry, second_face)

    first = _public_result(first_geometry, first_face, 0.5)
    second = _public_result(second_geometry, second_face, 0.5)
    first_validation = _public_validation(first, first_face)
    second_validation = _public_validation(second, second_face)

    assert _signature(first_geometry) == first_before
    assert _signature(second_geometry) == second_before
    assert first_validation["area_ratio"] == pytest.approx(1.0, rel=1.0e-10, abs=1.0e-10)
    assert second_validation["area_ratio"] == pytest.approx(1.0, rel=1.0e-10, abs=1.0e-10)
    _assert_public_edge_chains(first.mesh, first_domain)
    _assert_public_edge_chains(second.mesh, second_domain)
    _assert_no_hole_centroids(first.mesh, first_domain, first_face)
    _assert_no_hole_centroids(second.mesh, second_domain, second_face)

    first_counts = (len(first.mesh.nodes), len(first.mesh.quads), len(first.mesh.tris))
    second_counts = (len(second.mesh.nodes), len(second.mesh.quads), len(second.mesh.tris))
    assert first_counts == second_counts


def test_p03_public_h025_refines_and_follows_exact_hole_arcs() -> None:
    coarse_geometry, coarse_face, _ = _p03_geometry()
    fine_geometry, fine_face, fine_hole_edges = _p03_geometry()
    coarse_before = _signature(coarse_geometry)
    fine_before = _signature(fine_geometry)

    coarse = _public_result(coarse_geometry, coarse_face, 0.5)
    fine = _public_result(fine_geometry, fine_face, 0.25)
    coarse_validation = _public_validation(coarse, coarse_face)
    fine_validation = _public_validation(fine, fine_face)

    assert _signature(coarse_geometry) == coarse_before
    assert _signature(fine_geometry) == fine_before
    ratio = fine_validation["n_eq"] / coarse_validation["n_eq"]
    assert 3.2 <= ratio <= 5.0
    assert len(fine.mesh.nodes) > len(coarse.mesh.nodes)
    assert fine_validation["q4_count_fraction"] >= 0.75
    assert fine_validation["q4_area_fraction"] >= 0.75

    center = np.asarray((3.2, 2.4), dtype=float)
    radius = 0.9
    boundary_nodes: set[int] = set()
    maximum_chord = 0.0
    for edge_id in fine_hole_edges:
        chain = fine.mesh.nodes_of_edge[edge_id]
        boundary_nodes.update(int(node) for node in chain)
        points = np.asarray([fine.mesh.nodes[node][:2] for node in chain], dtype=float)
        distances = np.linalg.norm(points - center[None, :], axis=1)
        assert distances == pytest.approx(radius, abs=1.0e-10)
        if len(points) > 1:
            maximum_chord = max(maximum_chord, float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1))))

    assert maximum_chord <= 1.5 * 0.25 + 1.0e-12
    sagitta = radius - float(np.sqrt(max(radius * radius - 0.25 * maximum_chord * maximum_chord, 0.0)))
    for node_id, position in fine.mesh.nodes.items():
        if int(node_id) in boundary_nodes:
            continue
        distance = float(np.linalg.norm(np.asarray(position, dtype=float)[:2] - center))
        assert distance >= radius - sagitta - 1.0e-9


def _g3_hole_chord_profile(mesh, hole_edges) -> tuple:
    boundary_nodes: set[int] = set()
    maximum_chord = 0.0
    for edge_id in hole_edges:
        chain = mesh.nodes_of_edge[edge_id]
        boundary_nodes.update(int(node) for node in chain)
        points = np.asarray([mesh.nodes[node] for node in chain], dtype=float)
        if len(points) > 1:
            maximum_chord = max(
                maximum_chord,
                float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1))),
            )
    return boundary_nodes, maximum_chord


def test_p03_public_refines_with_target_size_and_respects_exact_circle_chords() -> None:
    # diagnostic reference (administrator measured, not hard-coded expectations):
    # h=.5  q4=233 t3=10  N_eq=238  chain lens [4,4,4,4] max_chord=.4658743
    # h=.25 q4=927 t3=12  N_eq=933  chain lens [7,7,7,7] max_chord=.23494715
    coarse_geometry, coarse_face, coarse_hole_edges = _p03_geometry()
    fine_geometry, fine_face, fine_hole_edges = _p03_geometry()
    coarse_before = _signature(coarse_geometry)
    fine_before = _signature(fine_geometry)

    coarse = _public_result(coarse_geometry, coarse_face, 0.5)
    fine = _public_result(fine_geometry, fine_face, 0.25)
    coarse_validation = _public_validation(coarse, coarse_face)
    fine_validation = _public_validation(fine, fine_face)

    assert _signature(coarse_geometry) == coarse_before
    assert _signature(fine_geometry) == fine_before

    coarse_n_eq = len(coarse.mesh.quads) + 0.5 * len(coarse.mesh.tris)
    fine_n_eq = len(fine.mesh.quads) + 0.5 * len(fine.mesh.tris)
    assert 3.2 <= fine_n_eq / coarse_n_eq <= 5.0

    for validation in (coarse_validation, fine_validation):
        assert validation["q4_count_fraction"] >= 0.75
        assert validation["q4_area_fraction"] >= 0.75
        assert validation["area_ratio"] == pytest.approx(1.0, rel=1.0e-10, abs=1.0e-10)

    center = np.asarray((3.2, 2.4, 0.0), dtype=float)
    radius = 0.9
    for mesh, hole_edges, h in (
        (coarse.mesh, coarse_hole_edges, 0.5),
        (fine.mesh, fine_hole_edges, 0.25),
    ):
        boundary_nodes, maximum_chord = _g3_hole_chord_profile(mesh, hole_edges)
        assert maximum_chord <= 1.5 * h + 1.0e-12
        for node_id in boundary_nodes:
            distance = float(np.linalg.norm(np.asarray(mesh.nodes[node_id], dtype=float) - center))
            assert distance == pytest.approx(radius, abs=1.0e-10)
        sagitta = radius - float(np.sqrt(max(0.0, radius * radius - 0.25 * maximum_chord * maximum_chord)))
        for node_id, position in mesh.nodes.items():
            if int(node_id) in boundary_nodes:
                continue
            distance = float(np.linalg.norm(np.asarray(position, dtype=float) - center))
            assert distance >= radius - sagitta - 1.0e-10


def _g3_public_profile(mesh, geometry, face) -> tuple:
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    edge_ids = [edge for edge, _ in domain.edge_uses]
    edge_ids.extend(edge for loop in domain.hole_edge_uses for edge, _ in loop)
    return (
        len(mesh.nodes),
        len(mesh.quads),
        len(mesh.tris),
        tuple(sorted(len(mesh.nodes_of_edge[edge_id]) for edge_id in edge_ids)),
    )


def test_p03_public_h025_deterministic_repeat() -> None:
    first_geometry, first_face, _ = _p03_geometry()
    second_geometry, second_face, _ = _p03_geometry()
    first = _public_result(first_geometry, first_face, 0.25)
    second = _public_result(second_geometry, second_face, 0.25)
    first_validation = _public_validation(first, first_face)
    second_validation = _public_validation(second, second_face)

    assert _g3_public_profile(first.mesh, first_geometry, first_face) == _g3_public_profile(
        second.mesh, second_geometry, second_face
    )
    for key in ("q4_count_fraction", "q4_area_fraction", "area_ratio"):
        assert first_validation[key] == pytest.approx(
            second_validation[key], rel=1.0e-12, abs=1.0e-12
        )
