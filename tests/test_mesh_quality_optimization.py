from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

from anygeometry import (
    EntityRef,
    GeometryModel,
    apply_imprint,
    plan_imprint,
    query_intersection,
)
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh
from anymesher import surface_mesh as surface_mesh_module
from anymesher.preparation import prepare_structural_closure
from anymesher.quality_v2 import assert_valid_mesh
from anymesher.seeding import edge_demand, solve_seeding
from anymesher.serialize import mesh_to_dict
from anymesher.surface_mesh import SurfaceMeshOptions, _triangle_quality, mesh_planar_surface


OUTER = np.asarray(((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)))
DIAGONAL = np.asarray(((0.5, 0.35), (3.5, 1.65)))


def _mesh(
    outer=OUTER,
    constraints=(DIAGONAL,),
    **options,
):
    diagnostics = {}
    mesh = mesh_planar_surface(
        outer,
        constraints=constraints,
        target_size=0.25,
        backend="python",
        diagnostics=diagnostics,
        **options,
    )
    return mesh, diagnostics["quality_optimization"]


def _contains_coordinate(mesh, expected) -> bool:
    expected = np.asarray(expected, dtype=np.float64)
    return bool(np.any(np.all(mesh.node_coordinates[:, :2] == expected, axis=1)))


def _edges(mesh) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(cell[index]), int(cell[(index + 1) % len(cell)]))))
        for cells in (mesh.triangle_connectivity, mesh.quad_connectivity)
        for cell in cells
        for index in range(len(cell))
    }


def _deterministic_mesh_payload(mesh):
    """Retain phase membership while excluding noncanonical elapsed samples."""

    def normalize(value):
        if isinstance(value, dict):
            return {
                key: (
                    sorted(item)
                    if key == "phase_seconds" and isinstance(item, dict)
                    else normalize(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(mesh_to_dict(mesh))


def _selected_mesh_digest(mesh) -> str:
    digest = sha256()
    for values in (
        mesh.node_coordinates,
        mesh.node_ids,
        mesh.triangle_connectivity,
        mesh.triangle_ids,
        mesh.quad_connectivity,
        mesh.quad_ids,
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _plate(model: GeometryModel, coordinates):
    points = [model.add_point(*coordinates) for coordinates in coordinates]
    face = model.add_plate(points)
    edges = tuple(oriented.edge for oriented in model.faces[face].loop)
    return face, edges


def test_segment_grid_is_a_deterministic_conservative_segment_aabb_index() -> None:
    base_segments = (
        (np.asarray((-4.0, -1.0)), np.asarray((4.0, -1.0))),
        (np.asarray((-0.5, -4.0)), np.asarray((-0.5, 4.0))),
        (np.asarray((2.5, 2.5)), np.asarray((4.0, 3.0))),
        (np.asarray((-4.0, 2.0)), np.asarray((-3.0, 2.25))),
    )
    base_query = (np.asarray((-2.0, -1.1)), np.asarray((2.0, -0.9)))
    padding = 0.05

    for angle in (0.0, 0.37):
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )
        segments = tuple(
            (first @ rotation.T, second @ rotation.T) for first, second in base_segments
        )
        first, second = (value @ rotation.T for value in base_query)
        grid = surface_mesh_module._SegmentGrid(segments, radius=0.2)
        candidates = grid.candidates_for_segment(first, second, padding)
        repeated = grid.candidates_for_segment(second, first, padding)

        query_lower = np.minimum(first, second) - padding
        query_upper = np.maximum(first, second) + padding
        exact_aabb = {
            index
            for index, (start, end) in enumerate(segments)
            if np.all(
                np.maximum(np.minimum(start, end), query_lower)
                <= np.minimum(np.maximum(start, end), query_upper)
            )
        }
        assert candidates == repeated
        assert candidates == tuple(sorted(set(candidates)))
        assert exact_aabb.issubset(candidates)


def test_diagonal_plate_quality_is_valid_bounded_and_deterministic() -> None:
    first, first_quality = _mesh(recombine=False)
    second, second_quality = _mesh(recombine=False)

    assert_valid_mesh(first)
    assert np.array_equal(first.node_coordinates, second.node_coordinates)
    assert np.array_equal(first.triangle_connectivity, second.triangle_connectivity)
    assert first_quality == second_quality
    assert first_quality["final_quality"]["invalid_element_count"] == 0
    assert first_quality["final_quality"]["max_aspect_ratio"] <= 5.0
    assert first_quality["target_met"] is True
    assert first_quality["rounds"] <= 2
    assert first_quality["added_points"] <= first_quality["point_budget"]
    assert first_quality["point_budget"] <= 0.5 * first_quality["initial_generated_points"]
    for point in (*OUTER, *DIAGONAL):
        assert _contains_coordinate(first, point)
    diagonal_rows = [
        int(np.flatnonzero(np.all(first.node_coordinates[:, :2] == point, axis=1))[0])
        for point in DIAGONAL
    ]
    assert tuple(sorted(diagonal_rows)) in _edges(first)


@pytest.mark.parametrize("angle", (0.0, 0.37, -0.81))
@pytest.mark.parametrize("reverse", (False, True))
def test_rotated_and_reversed_diagonal_variants_preserve_quality(angle, reverse) -> None:
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    outer = OUTER @ rotation.T
    diagonal = DIAGONAL @ rotation.T
    if reverse:
        outer = outer[::-1]
        diagonal = diagonal[::-1]
    mesh, quality = _mesh(outer, (diagonal,), recombine=False)

    assert_valid_mesh(mesh)
    assert quality["final_quality"]["invalid_element_count"] == 0
    assert quality["final_quality"]["max_aspect_ratio"] <= 5.0
    for point in (*outer, *diagonal):
        assert _contains_coordinate(mesh, point)


def test_hole_explicit_point_quadratic_and_recombination_keep_protected_nodes() -> None:
    hole = np.asarray(((1.5, 0.75), (2.5, 0.75), (2.5, 1.25), (1.5, 1.25)))
    explicit = np.asarray(((1.0, 1.0), (3.0, 1.0)))
    diagnostics = {}
    mesh = mesh_planar_surface(
        OUTER,
        holes=(hole,),
        interior_points=explicit,
        target_size=0.25,
        backend="python",
        recombine=True,
        order="quadratic",
        diagnostics=diagnostics,
    )

    assert_valid_mesh(mesh)
    for point in (*OUTER, *hole, *explicit):
        assert _contains_coordinate(mesh, point)
    quality = diagnostics["quality_optimization"]
    assert quality["final_quality"]["invalid_element_count"] == 0
    assert quality["rounds"] <= 2


def test_unavoidable_acute_corner_returns_best_valid_mesh_with_diagnostics() -> None:
    acute = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.02, 0.10)))
    mesh, quality = _mesh(acute, (), recombine=False)

    assert_valid_mesh(mesh)
    final = quality["final_quality"]
    assert final["invalid_element_count"] == 0
    assert final["poor_element_ids"] == sorted(final["poor_element_ids"])
    assert quality["target_met"] is False
    assert quality["budget_exhausted"] is True


def test_quality_cancellation_stops_at_a_safe_boundary() -> None:
    phases = []

    def cancel(phase: str) -> None:
        phases.append(phase)
        if phase == "native surface quality optimization start":
            raise MeshError("cancelled quality pass")

    with pytest.raises(MeshError, match="cancelled quality pass"):
        mesh_planar_surface(
            OUTER,
            constraints=(DIAGONAL,),
            target_size=0.25,
            backend="python",
            cancellation_check=cancel,
        )
    assert phases[-1] == "native surface quality optimization start"


@pytest.mark.skipif(
    not surface_mesh_module._native_cpp.COMPILED_QUALITY_PIPELINE_AVAILABLE,
    reason="compiled quality pipeline is unavailable",
)
@pytest.mark.parametrize(
    ("points", "triangles"),
    (
        (
            np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (5.0, 0.0))
            ),
            np.asarray(((0, 1, 2), (1, 3, 2))),
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0e-13, 1.0e-15))),
            np.asarray(((0, 2, 1),)),
        ),
        (
            np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
            ),
            np.asarray(((2, 3, 0), (3, 1, 0))),
        ),
        (
            np.asarray(
                (
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.8660254037844386, 0.5),
                    (3.0, 0.0),
                    (4.0, 0.0),
                    (2.133974596215561, 0.5),
                )
            ),
            np.asarray(((0, 1, 2), (3, 4, 5))),
        ),
    ),
)
def test_triangle_quality_compiled_matches_python_report_score_and_aspects(
    monkeypatch, points, triangles
) -> None:
    settings = SurfaceMeshOptions(
        prefer_quality_policy=True,
        max_element_growth=1.25,
    )
    monkeypatch.setattr(
        surface_mesh_module._native_cpp,
        "COMPILED_QUALITY_PIPELINE_AVAILABLE",
        False,
    )
    python_report, python_score, python_aspects = _triangle_quality(
        points, triangles, settings
    )
    monkeypatch.setattr(
        surface_mesh_module._native_cpp,
        "COMPILED_QUALITY_PIPELINE_AVAILABLE",
        True,
    )
    native_report, native_score, native_aspects = _triangle_quality(
        points, triangles, settings
    )

    assert native_report.keys() == python_report.keys()
    for key in native_report:
        if isinstance(native_report[key], float):
            assert native_report[key] == pytest.approx(
                python_report[key], rel=1.0e-12, abs=1.0e-12
            )
        else:
            assert native_report[key] == python_report[key]
    assert native_score[:3] == python_score[:3]
    assert native_score[3:] == pytest.approx(
        python_score[3:], rel=1.0e-12, abs=1.0e-12
    )
    np.testing.assert_array_equal(native_aspects, python_aspects)


@pytest.mark.skipif(
    not surface_mesh_module._native_cpp.COMPILED_QUALITY_PIPELINE_AVAILABLE,
    reason="compiled quality pipeline is unavailable",
)
def test_compiled_triangle_quality_preserves_selected_mesh(monkeypatch) -> None:
    diagnostics = {}
    monkeypatch.setattr(
        surface_mesh_module._native_cpp,
        "COMPILED_QUALITY_PIPELINE_AVAILABLE",
        False,
    )
    python_mesh = mesh_planar_surface(
        OUTER,
        constraints=(DIAGONAL,),
        target_size=0.25,
        backend="python",
        recombine=False,
        diagnostics=diagnostics,
    )
    python_quality = diagnostics["quality_optimization"]
    diagnostics = {}
    monkeypatch.setattr(
        surface_mesh_module._native_cpp,
        "COMPILED_QUALITY_PIPELINE_AVAILABLE",
        True,
    )
    native_mesh = mesh_planar_surface(
        OUTER,
        constraints=(DIAGONAL,),
        target_size=0.25,
        backend="python",
        recombine=False,
        diagnostics=diagnostics,
    )
    native_quality = diagnostics["quality_optimization"]

    np.testing.assert_array_equal(
        native_mesh.node_coordinates, python_mesh.node_coordinates
    )
    np.testing.assert_array_equal(
        native_mesh.triangle_connectivity, python_mesh.triangle_connectivity
    )
    np.testing.assert_array_equal(
        native_mesh.quad_connectivity, python_mesh.quad_connectivity
    )
    assert _selected_mesh_digest(native_mesh) == _selected_mesh_digest(python_mesh)
    for stage in ("initial_quality", "final_quality"):
        assert native_quality[stage].keys() == python_quality[stage].keys()
        for key, native_value in native_quality[stage].items():
            python_value = python_quality[stage][key]
            if isinstance(native_value, float):
                assert native_value == pytest.approx(
                    python_value, rel=1.0e-12, abs=1.0e-12
                )
            else:
                assert native_value == python_value
    assert native_quality["candidates"][0]["score"] == pytest.approx(
        python_quality["candidates"][0]["score"],
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_present_compiled_triangle_quality_rejects_malformed_batches(
    monkeypatch,
) -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    triangles = np.asarray(((0, 1, 2),))
    monkeypatch.setattr(
        surface_mesh_module._native_cpp,
        "COMPILED_QUALITY_PIPELINE_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        surface_mesh_module._native_cpp,
        "native_element_quality",
        lambda *_args: np.zeros((1, 5)),
    )
    with pytest.raises(MeshError, match="malformed output"):
        _triangle_quality(points, triangles)


def test_compiled_triangle_incidence_rejects_repeated_cell_attachment(
    monkeypatch,
) -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    monkeypatch.setattr(
        surface_mesh_module._native_cpp,
        "triangle_edge_incidence",
        lambda _triangles: np.asarray(
            (
                (0, 1, 0, -1),
                (0, 2, 0, 0),
                (0, 3, 1, -1),
                (1, 2, 0, -1),
                (2, 3, 1, -1),
            ),
            dtype=np.int64,
        ),
    )
    with pytest.raises(MeshError, match="repeats an attachment"):
        _triangle_quality(points, triangles)


def test_three_plate_connect_quality_and_shared_identity_are_deterministic() -> None:
    geometry = GeometryModel()
    support, support_edges = _plate(
        geometry,
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
    )
    edge_wall = geometry.extrude((support_edges[0],), (0.0, 0.0, 1.0))[0]
    diagonal_points = geometry.add_points(((0.5, 0.35, 0.0), (3.5, 1.65, 0.0)))
    diagonal = geometry.add_line(*diagonal_points)
    top_end, top_start = geometry.add_points(((3.5, 1.80, 1.0), (0.5, 0.35, 1.0)))
    diagonal_wall = geometry.add_face(
        (
            diagonal,
            geometry.add_line(diagonal_points[1], top_end),
            geometry.add_line(top_end, top_start),
            geometry.add_line(top_start, diagonal_points[0]),
        )
    )
    geometry.add_sheet((support,))
    geometry.add_sheet((edge_wall,))
    geometry.add_sheet((diagonal_wall,))
    intersection = query_intersection(
        geometry,
        geometry.handle("face", support),
        geometry.handle("face", diagonal_wall),
    )
    application = apply_imprint(
        geometry,
        plan_imprint(geometry, intersection, policy="connect"),
        policy="connect",
    )

    first = generate_hybrid_mesh(
        geometry,
        target_size=0.25,
        strategy="auto",
        structured_options={"max_element_growth": 1.5},
        native_backend="python",
    )
    second = generate_hybrid_mesh(
        geometry,
        target_size=0.25,
        strategy="auto",
        structured_options={"max_element_growth": 1.5},
        native_backend="python",
    )

    assert application.face_intersection is not None
    assert application.face_intersection.edge.id == diagonal
    assert _deterministic_mesh_payload(first) == _deterministic_mesh_payload(second)
    for edge_id, expected_wall in (
        (support_edges[0], edge_wall),
        (diagonal, diagonal_wall),
    ):
        shared = set(first.nodes_on(EntityRef("edge", edge_id)))
        assert shared
        incident_faces = {
            geometry.face_uses[face_use_id].face_id
            for face_use_id in geometry.face_uses_using_edge(edge_id)
        }
        assert expected_wall in incident_faces
        assert len(incident_faces) >= 2
        assert all(
            shared <= set(first.nodes_on(EntityRef("face", face_id)))
            for face_id in incident_faces
        )
    face_diagnostics = first.hybrid_diagnostics["triangulation_backend_by_face"]
    assert set(face_diagnostics) == set(geometry.faces)
    repair = first.hybrid_diagnostics.get("junction_growth_repair")
    if repair is not None:
        assert repair["attempted"] is True
        assert repair["committed"] is True
        assert repair["junction_node_pairs"]
        assert repair["moved_node_ids"]
        assert repair["initial_quality"]["growth_violation_count"] == 1
        assert repair["final_quality"]["growth_violation_count"] == 0
        assert repair["final_quality"]["maximum_adjacent_element_growth"] < repair[
            "initial_quality"
        ]["maximum_adjacent_element_growth"]
        assert repair["final_quality"]["maximum_aspect_ratio"] <= 5.0
    else:
        assert all(
            item["quality_optimization"]["final_quality"][
                "max_element_growth"
            ]
            <= 1.5
            for diagnostic in face_diagnostics.values()
            for item in diagnostic["working_face_diagnostics"]
        )


def test_two_level_plate_intersections_refine_thin_wall_strips_compatibly() -> None:
    geometry = GeometryModel()
    lower, _ = _plate(
        geometry,
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (-1.0, 2.0, 0.0)),
    )
    upper, _ = _plate(
        geometry,
        ((-1.0, 0.0, 0.1), (1.0, 0.0, 0.1), (1.0, 2.0, 0.1), (-1.0, 2.0, 0.1)),
    )
    wall, _ = _plate(
        geometry,
        ((0.0, 0.0, -0.2), (0.0, 2.0, -0.2), (0.0, 2.0, 0.8), (0.0, 0.0, 0.8)),
    )
    geometry.add_sheet((lower,))
    geometry.add_sheet((upper,))
    geometry.add_sheet((wall,))

    prepared, report = prepare_structural_closure(geometry)
    assert report.face_connections == 2
    wall_faces = report.source_to_working_faces[wall]
    seeding = solve_seeding(prepared, target_size=0.25)
    for face_id in wall_faces:
        face = prepared.faces[face_id]
        if len(face.corners) != 4:
            continue
        axis_steps = []
        sides = face.sides()
        for first, second in ((0, 2), (1, 3)):
            side_steps = []
            for side in (sides[first], sides[second]):
                demand = sum(
                    edge_demand(prepared, item.edge, seeding.size_field)
                    for item in side
                )
                divisions = sum(seeding[item.edge] for item in side)
                side_steps.append(demand / divisions)
            axis_steps.append(max(side_steps))
        assert max(axis_steps) <= 1.7320508075688772 * min(axis_steps) + 1.0e-12

    first = generate_hybrid_mesh(
        geometry,
        target_size=0.25,
        strategy="native",
        native_backend="python",
    )
    second = generate_hybrid_mesh(
        geometry,
        target_size=0.25,
        strategy="native",
        native_backend="python",
    )
    assert _deterministic_mesh_payload(first) == _deterministic_mesh_payload(second)
    face_diagnostics = first.hybrid_diagnostics["triangulation_backend_by_face"]
    wall_quality = [
        item["quality_optimization"]["final_quality"]
        for item in face_diagnostics[wall]["working_face_diagnostics"]
    ]
    assert all(
        item["actual_backend"] == "python"
        for item in face_diagnostics[wall]["working_face_diagnostics"]
    )
    assert max(item["max_aspect_ratio"] for item in wall_quality) <= 5.0
    assert min(item["min_angle"] for item in wall_quality) >= 20.0
    assert max(item["max_element_growth"] for item in wall_quality) <= 1.5
    assert all(item["repair_element_ids"] == [] for item in wall_quality)


def test_anyfem_exact_floating_plate_and_diagonal_extrusion_meshes_automatically() -> None:
    geometry = GeometryModel()
    first, second, third, fourth = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (2.0, 2.0, 0.0))
    )
    support = geometry.add_plate((first, second, fourth, third))
    fifth, sixth, seventh, eighth = geometry.add_points(
        ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 1.5, 0.5), (1.5, 1.5, 0.5))
    )
    floating = geometry.add_plate((fifth, sixth, eighth, seventh))
    diagonal = geometry.add_line(second, third)
    before_extrusion = set(geometry.faces)
    geometry.extrude((diagonal,), (0.0, 0.0, 1.0))
    wall, = set(geometry.faces) - before_extrusion
    geometry.add_sheet((support,))
    geometry.add_sheet((floating,))

    assert len(geometry.vertices) == 10
    assert len(geometry.edges) == 12
    assert len(geometry.faces) == 3
    assert len(geometry.sheets) == 2
    options = {
        "target_size": 0.25,
        "strategy": "auto",
        "native_backend": "python",
        "structured_options": {
            "preference": "balanced",
            "quality_policy": {
                "minimum_scaled_jacobian": 0.10,
                "maximum_aspect_ratio": 5.0,
                "minimum_angle": 20.0,
                "maximum_angle": 160.0,
                "maximum_warpage": 0.10,
            },
        },
    }
    first_mesh = generate_hybrid_mesh(geometry, **options)
    second_mesh = generate_hybrid_mesh(geometry, **options)

    assert _deterministic_mesh_payload(first_mesh) == _deterministic_mesh_payload(
        second_mesh
    )
    assert first_mesh.automatic_intersections == 2
    assert first_mesh.declared_plate_junction_edges
