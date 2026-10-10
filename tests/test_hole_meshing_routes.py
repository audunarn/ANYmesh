"""Regressions for the two hole routes that failed in the 2026-10-09 S0 spike.

A neutral trimmed hole (``anygeometry.punch_hole``) in a plain strip could not
be meshed for qualified S3.  Two independent causes stacked:

* A face created without a surface carries a topology Coons marker.  Native
  meshing used its unit parameter square as the chart, so a 2:0.5 strip was
  meshed 4:1 stretched and qualified-S3 admission (correctly) refused the
  resulting slivers.  An exactly affine Coons face now uses its Plane chart.
* Native selection ranks a boundary-collar candidate first for its
  alignment even when its recombined cells break the element-growth limit.
  The structured route's gate then rejected every candidate the fallback
  chain offered.  The chain now ends with one native candidate whose
  selection respects that limit.  Meshes the chain already accepted are
  unchanged.

The butterfly decomposition (``anymesher.punch_circular_hole``) gave its
patches no surface.  An arc-bounded topology Coons patch has neither analytic
nor bilinear support, so structural intersection planning refused the model.
The patches now carry the parent's planar support.
"""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    ConnectionIntent,
    GeometryModel,
    IntersectionBatchPolicy,
    Plane,
    plan_intersections,
    punch_hole,
)
from anygeometry.entities import OrientedEdge
from anygeometry.surfaces import CoonsSurface
from anymesher import hybrid, punch_circular_hole
from anymesher.hybrid import _structured_quality_report, generate_hybrid_mesh_result
from anymesher.structured import StructuredMeshingOptions
from anymesher.surface_mesh import SurfaceMeshOptions, mesh_planar_surface

# The keyword set ANYfem's Project.generate_mesh passes for a plain plate model.
ANYFEM_ROUTE = dict(
    structural_preparation={
        "automatic_face_connections": False,
        "automatic_member_connections": False,
        "automatic_member_sheet_connections": False,
        "declare_missing_owners": True,
    },
    mutation_policy="working_copy",
    qualified_s3=True,
    structured_options=StructuredMeshingOptions(),
)
S0_SIZES = (0.125, 0.05, 0.02)


def _quad(geometry: GeometryModel, points, *, plate: bool) -> int:
    vertices = geometry.add_points(points)
    if plate:
        return geometry.add_plate(vertices)
    return geometry.add_face(geometry.add_polyline(vertices, close=True))


def _neutral_strip(*, plate: bool) -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    face = _quad(
        geometry,
        ((-1.0, -0.25, 0.0), (1.0, -0.25, 0.0), (1.0, 0.25, 0.0), (-1.0, 0.25, 0.0)),
        plate=plate,
    )
    punch_hole(geometry, face, (0.0, 0.0, 0.0), 0.05)
    return geometry, face


def _strip_of_three(geometry: GeometryModel, *, plate: bool) -> list[int]:
    """[-1,-0.25], [-0.25,0.25], [0.25,1] x [-0.25,0.25] sharing their dividers."""

    xs = (-1.0, -0.25, 0.25, 1.0)
    vertices = {
        (i, j): geometry.add_point(x, y, 0.0)
        for j, y in enumerate((-0.25, 0.25))
        for i, x in enumerate(xs)
    }
    if plate:
        return [
            geometry.add_plate(
                [vertices[i, 0], vertices[i + 1, 0], vertices[i + 1, 1], vertices[i, 1]]
            )
            for i in range(3)
        ]
    horizontal = {
        (i, j): geometry.add_line(vertices[i, j], vertices[i + 1, j])
        for j in range(2)
        for i in range(3)
    }
    vertical = [geometry.add_line(vertices[i, 0], vertices[i, 1]) for i in range(4)]
    return [
        geometry.add_face([horizontal[i, 0], vertical[i + 1], horizontal[i, 1], vertical[i]])
        for i in range(3)
    ]


def _butterfly_strip(*, plate: bool) -> tuple[GeometryModel, list[int], list[int]]:
    geometry = GeometryModel()
    faces = _strip_of_three(geometry, plate=plate)
    patches, arcs = punch_circular_hole(geometry, faces[1], (0.0, 0.0, 0.0), 0.05)
    return geometry, patches, arcs


def _ring_pins(geometry: GeometryModel, divisions: int = 8) -> dict[int, int]:
    """Arcs and spokes pinned to equal divisions, as in the S0 O-grid control."""

    pins = {}
    for edge_id, edge in geometry.edges.items():
        radii = [
            float(np.linalg.norm(np.asarray(geometry.vertex_position(vertex))[:2]))
            for vertex in (edge.start, edge.end)
        ]
        if any(abs(radius - 0.05) < 1.0e-9 for radius in radii):
            pins[edge_id] = divisions
    return pins


def _minimum_triangle_angle(mesh) -> float:
    angles = []
    for connectivity in mesh.tris.values():
        corners = np.asarray([mesh.nodes[node] for node in connectivity[:3]])
        for index in range(3):
            first = corners[(index + 1) % 3] - corners[index]
            second = corners[(index + 2) % 3] - corners[index]
            cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
            angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return min(angles, default=60.0)


# ----------------------------------------------------------------------
# Defect 1a: an exactly affine topology Coons face is meshed in its Plane chart
# ----------------------------------------------------------------------
def test_affine_topology_coons_face_has_the_plane_of_its_own_coordinates() -> None:
    geometry = GeometryModel()
    rectangle = _quad(
        geometry,
        ((-1.0, -0.25, 0.0), (1.0, -0.25, 0.0), (1.0, 0.25, 0.0), (-1.0, 0.25, 0.0)),
        plate=False,
    )
    sheared = _quad(
        geometry,
        ((0.0, 2.0, 1.0), (3.0, 2.0, 1.0), (4.0, 3.0, 2.0), (1.0, 3.0, 2.0)),
        plate=False,
    )
    assert isinstance(geometry.faces[rectangle].surface, CoonsSurface)

    for face in (rectangle, sheared):
        support = hybrid._native_planar_support(geometry, face)
        assert isinstance(support, Plane)
        for u, v in ((0.0, 0.0), (0.3, 0.7), (1.0, 1.0), (0.85, 0.1)):
            assert np.allclose(
                support.evaluate(u, v), geometry.face_point(face, u, v), atol=1.0e-12
            )
    support = hybrid._native_planar_support(geometry, rectangle)
    assert np.allclose(support.u_vector, (2.0, 0.0, 0.0))
    assert np.allclose(support.v_vector, (0.0, 0.5, 0.0))


def test_side_of_collinear_edges_keeps_the_affine_plane_chart() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0, 0, 0), (0.7, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))
    )
    face = geometry.add_face_from_loop(
        [OrientedEdge(edge, True) for edge in geometry.add_polyline(points, close=True)],
        (0, 2, 3, 4),
    )
    assert len(geometry.faces[face].sides()[0]) == 2

    support = hybrid._native_planar_support(geometry, face)

    assert isinstance(support, Plane)
    assert np.allclose(support.evaluate(0.5, 0.5), geometry.face_point(face, 0.5, 0.5))


def test_non_affine_or_curved_coons_faces_keep_their_parameter_chart() -> None:
    geometry = GeometryModel()
    trapezoid = _quad(
        geometry, ((0, 0, 0), (3, 0, 0), (2, 1, 0), (1, 1, 0)), plate=False
    )
    points = geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    via = geometry.add_point(1.0, 1.3, 0.0)
    curved = geometry.add_face(
        [
            geometry.add_line(points[0], points[1]),
            geometry.add_line(points[1], points[2]),
            geometry.add_arc(points[2], via, points[3]),
            geometry.add_line(points[3], points[0]),
        ]
    )
    plate = _quad(geometry, ((5, 0, 0), (6, 0, 0), (6, 1, 0), (5, 1, 0)), plate=True)

    assert hybrid._native_planar_support(geometry, trapezoid) is None
    assert hybrid._native_planar_support(geometry, curved) is None
    assert hybrid._native_planar_support(geometry, plate) is geometry.faces[plate].surface


def test_neutral_hole_coons_strip_meshes_exactly_like_its_plate() -> None:
    meshes = []
    for plate in (False, True):
        geometry, face = _neutral_strip(plate=plate)
        meshes.append(
            generate_hybrid_mesh_result(geometry, target_size=0.05, face_ids=[face]).mesh
        )
    coons, plane = meshes

    assert (len(coons.quads), len(coons.tris)) == (len(plane.quads), len(plane.tris))
    first, second = coons.node_positions(), plane.node_positions()
    assert first.shape == second.shape
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    # Both owners evaluate the same affine map; only the last bits differ.
    assert distances.min(axis=1).max() <= 1.0e-12
    assert distances.min(axis=0).max() <= 1.0e-12


@pytest.mark.parametrize("size", S0_SIZES)
def test_neutral_hole_in_coons_strip_is_admitted_for_qualified_s3(size: float) -> None:
    geometry, face = _neutral_strip(plate=False)

    mesh = generate_hybrid_mesh_result(
        geometry, target_size=size, face_ids=[face], qualified_s3=True
    ).mesh

    assert mesh.structural_preparation["qualified_s3"]["status"] == "ADMITTED"
    assert _minimum_triangle_angle(mesh) >= 15.0


# ----------------------------------------------------------------------
# Defect 1b: the fallback chain ends with a growth-limited native candidate
# ----------------------------------------------------------------------
GROWTH_POLICY = SurfaceMeshOptions(
    target_size=0.125, recombine=True, min_angle=15.0, prefer_quality_policy=True
)


def test_growth_limited_selection_skips_a_collar_that_breaks_growth() -> None:
    outer = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=float)
    diagnostics: dict = {}

    core = mesh_planar_surface(
        outer,
        (),
        options=GROWTH_POLICY,
        diagnostics=diagnostics,
        _respect_growth_limit=True,
    )

    optimization = diagnostics["quality_optimization"]
    collars = [
        item for item in optimization["candidates"]
        if item["strategy"] == "outer_boundary_collar"
    ]
    # The square's collar corners exceed the 1.5 limit; the baseline does not.
    assert collars and collars[0]["published_growth"]["elements_above_maximum"]
    assert not collars[0]["quality_eligible"]
    assert optimization["selected_strategy"] == "staggered_chart"
    assert optimization["final_quality"]["max_element_growth"] <= 1.5
    assert core.num_quads


def test_default_selection_still_prefers_the_aligned_collar() -> None:
    outer = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=float)
    diagnostics: dict = {}

    mesh_planar_surface(outer, (), options=GROWTH_POLICY, diagnostics=diagnostics)

    optimization = diagnostics["quality_optimization"]
    assert optimization["selected_strategy"] == "outer_boundary_collar"
    assert optimization["final_quality"]["max_element_growth"] > 1.5


@pytest.mark.parametrize(
    ("plate", "size"),
    # The literal S0 strip at every size; its add_plate twin at the coarse two.
    [(False, size) for size in S0_SIZES] + [(True, size) for size in S0_SIZES[:2]],
)
def test_neutral_hole_strip_meshes_through_the_anyfem_route(plate: bool, size: float) -> None:
    geometry, _face = _neutral_strip(plate=plate)

    mesh = generate_hybrid_mesh_result(geometry, target_size=size, **ANYFEM_ROUTE).mesh

    assert mesh.structural_preparation["qualified_s3"]["status"] == "ADMITTED"
    assert _minimum_triangle_angle(mesh) >= 15.0
    assert _structured_quality_report(mesh, StructuredMeshingOptions())["accepted"]
    selected = mesh.hybrid_diagnostics["growth_limited_native_candidate"]
    assert selected["rejected_fallback_quality"]["growth_violation_count"]
    assert selected["accepted_quality"]["accepted"]


# ----------------------------------------------------------------------
# Defect 2: butterfly patches carry their parent's planar support
# ----------------------------------------------------------------------
@pytest.mark.parametrize("plate", (False, True), ids=("add_face", "add_plate"))
def test_butterfly_patches_carry_a_covering_plane_like_their_parent(plate: bool) -> None:
    geometry = GeometryModel()
    parent = _quad(
        geometry, ((0, 0, 0), (2, 0, 0), (2, 1.5, 0), (0, 1.5, 0)), plate=plate
    )
    parent_surface = geometry.faces[parent].surface
    parent_normal = np.asarray(geometry.face_normal(parent, 0.5, 0.5))

    patches, arcs = punch_circular_hole(geometry, parent, (1.0, 0.75, 0.0), 0.3)

    for patch in patches:
        support = geometry.faces[patch].surface
        assert isinstance(support, Plane)
        if plate:
            assert support is parent_surface
        assert np.allclose(geometry.face_normal(patch, 0.5, 0.5), parent_normal)
        boundary = [
            geometry.vertex_position(vertex)
            for item in geometry.faces[patch].loop
            for vertex in (geometry.edges[item.edge].start, geometry.edges[item.edge].end)
        ]
        boundary += [geometry.sample_edge(arc, np.asarray([0.5]))[0] for arc in arcs]
        for point in boundary:
            _projected, _uv, distance = geometry.project_to_face(patch, point)
            if np.linalg.norm(np.asarray(point)[:2] - (1.0, 0.75)) < 0.3 + 1.0e-9:
                continue
            assert distance <= 1.0e-9


def test_butterfly_patches_follow_a_reversed_parent_orientation() -> None:
    geometry = GeometryModel()
    parent = _quad(
        geometry, ((0, 0, 0), (0, 1.5, 0), (2, 1.5, 0), (2, 0, 0)), plate=False
    )
    parent_normal = np.asarray(geometry.face_normal(parent, 0.5, 0.5))
    assert parent_normal[2] < 0.0

    patches, _arcs = punch_circular_hole(geometry, parent, (1.0, 0.75, 0.0), 0.3)

    for patch in patches:
        assert np.allclose(geometry.face_normal(patch, 0.5, 0.5), parent_normal)


@pytest.mark.parametrize("plate", (False, True), ids=("add_face", "add_plate"))
def test_butterfly_strip_is_accepted_by_structural_intersection_planning(plate: bool) -> None:
    geometry, _patches, _arcs = _butterfly_strip(plate=plate)

    plan = plan_intersections(
        geometry,
        tuple(geometry.handle("face", face) for face in sorted(geometry.faces)),
        policy=IntersectionBatchPolicy(ConnectionIntent.CONNECT),
    )

    assert plan.arrangements


@pytest.mark.parametrize("plate", (False, True), ids=("add_face", "add_plate"))
def test_pinned_butterfly_strip_meshes_as_quads_through_the_anyfem_route(plate: bool) -> None:
    geometry, _patches, _arcs = _butterfly_strip(plate=plate)

    mesh = generate_hybrid_mesh_result(
        geometry, target_size=0.5 / 8, overrides=_ring_pins(geometry), **ANYFEM_ROUTE
    ).mesh

    assert len(mesh.quads) == 448
    assert not mesh.tris


# ----------------------------------------------------------------------
# The literal S0 reproducers through ANYfem, when ANYfem is installed
# ----------------------------------------------------------------------
def _anyfem_project(name: str):
    anyfem = pytest.importorskip(
        "anyfem",
        reason=(
            "ANYfem consumer check (owner: ANYmesh); ANYmesh CI installs only "
            "ANYgeometry. Remove the skip when CI installs ANYfem."
        ),
    )
    project = anyfem.Project(name=name)
    project.add_material(anyfem.steel("S355", 0.010))
    project.add_plate_section("plate", thickness=0.010, material="S355")
    return project


def test_anyfem_meshes_the_neutral_hole_strip() -> None:
    project = _anyfem_project("neutral")
    geometry = project.geometry
    face = _quad(
        geometry,
        ((-1.0, -0.25, 0.0), (1.0, -0.25, 0.0), (1.0, 0.25, 0.0), (-1.0, 0.25, 0.0)),
        plate=False,
    )
    punch_hole(geometry, face, (0.0, 0.0, 0.0), 0.05)
    project.assign_plate(face, "plate")

    mesh = project.generate_mesh(0.05)

    assert mesh.quads and mesh.tris


def test_anyfem_meshes_the_pinned_butterfly_strip() -> None:
    project = _anyfem_project("butterfly")
    geometry = project.geometry
    faces = _strip_of_three(geometry, plate=False)
    punch_circular_hole(geometry, faces[1], (0.0, 0.0, 0.0), 0.05)
    for face in geometry.faces:
        project.assign_plate(face, "plate")

    mesh = project.generate_mesh(0.5 / 8, overrides=_ring_pins(geometry))

    assert len(mesh.quads) == 448
    assert not mesh.tris
