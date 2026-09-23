"""CH8 conical quad-first public linear contract (analytic isometric conical chart)."""

from __future__ import annotations

import numpy as np
import pytest

from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state
from test_curved_native_qualification import _model_face

from anygeometry import GeometryModel
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.domain import ConicalQuadDomain
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported


def _generate(model, face_ids, target_size, *, order="linear", cancellation_check=None):
    return generate_hybrid_mesh_result(
        model,
        face_ids=tuple(face_ids),
        target_size=float(target_size),
        strategy="native",
        native_backend="python",
        recombine=True,
        order=order,
        quad_options=QuadMeshingOptions(),
        cancellation_check=cancellation_check,
    )


def _n_eq(mesh) -> float:
    return float(len(mesh.quads)) + 0.5 * float(len(mesh.tris))


def _support_residual(mesh, surface) -> float:
    worst = 0.0
    for point in mesh.nodes.values():
        point = np.asarray(point, dtype=float)
        uv = surface.local_uv(tuple(point))
        worst = max(worst, float(np.linalg.norm(np.asarray(surface.evaluate(*uv), dtype=float) - point)))
    return worst


def _shoelace(points) -> float:
    total = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return abs(0.5 * total)


def _element_chart_area(domain, mesh, node_ids) -> float:
    chart = tuple(domain.project(mesh.nodes[int(node)]) for node in node_ids)
    return _shoelace(chart)


def _published_boundary_chart_area(domain, model, mesh, face) -> float:
    loop_nodes: list[int] = []
    for use in model.faces[int(face)].loop:
        chain = tuple(int(node) for node in mesh.nodes_of_edge[int(use.edge)])
        if not bool(use.forward):
            chain = tuple(reversed(chain))
        if loop_nodes:
            assert loop_nodes[-1] == chain[0]
            loop_nodes.extend(chain[1:])
        else:
            loop_nodes.extend(chain)
    assert loop_nodes[-1] == loop_nodes[0]
    chart = tuple(domain.project(mesh.nodes[node]) for node in loop_nodes[:-1])
    return _shoelace(chart)


def _assert_linear_shell_topology(mesh, face: int) -> None:
    assert mesh.quads
    assert set(mesh.elements_of_face) == {face}
    assert set(mesh.elements_of_face[face]) == set(mesh.quads) | set(mesh.tris)
    assert all(len(connectivity) == 4 for connectivity in mesh.quads.values())
    assert all(len(connectivity) == 3 for connectivity in mesh.tris.values())


def _assert_full_nodes_of_edge_chains(model, mesh, face: int) -> None:
    face_nodes: set[int] = set()
    for element_id in mesh.elements_of_face[face]:
        face_nodes.update(mesh.quads[element_id] if element_id in mesh.quads else mesh.tris[element_id])
    for use in model.faces[face].loop:
        edge_id = int(use.edge)
        assert edge_id in mesh.nodes_of_edge
        chain = tuple(mesh.nodes_of_edge[edge_id])
        assert len(chain) >= 2
        assert len(chain) == len(set(chain))
        assert set(chain) <= face_nodes
        first = np.asarray(model.vertex_position(model.edges[edge_id].start), dtype=float)
        last = np.asarray(model.vertex_position(model.edges[edge_id].end), dtype=float)
        assert np.allclose(np.asarray(mesh.nodes[chain[0]], dtype=float), first, atol=1.0e-9)
        assert np.allclose(np.asarray(mesh.nodes[chain[-1]], dtype=float), last, atol=1.0e-9)


def test_ch8_conical_domain_roundtrip_binds_model_revision_face() -> None:
    model, face, surface = _model_face("cone")
    before = _persistent_state(model)
    domain = ConicalQuadDomain.from_geometry(model, face)
    assert domain.model_id == str(model.model_id)
    assert domain.revision == int(model.revision)
    assert domain.face_id == int(face)
    assert domain.area > 0.0
    assert domain.outer_area > 0.0

    for vertex_id in domain.vertex_ids:
        world = tuple(np.asarray(model.vertex_position(vertex_id), dtype=float))
        assert np.allclose(domain.lift(domain.project(world)), world, atol=1.0e-11)

    mid = tuple(np.asarray(surface.evaluate(0.5, 0.5), dtype=float))
    assert np.allclose(domain.lift(domain.project(mid)), mid, atol=1.0e-9)

    assert _support_residual_of_points([mid], surface) <= 1.0e-10
    assert _persistent_state(model) == before


def _support_residual_of_points(points, surface) -> float:
    worst = 0.0
    for point in points:
        point = np.asarray(point, dtype=float)
        uv = surface.local_uv(tuple(point))
        worst = max(worst, float(np.linalg.norm(np.asarray(surface.evaluate(*uv), dtype=float) - point)))
    return worst


def test_ch8_conical_linear_refines_with_target_size() -> None:
    coarse_model, coarse_face, surface = _model_face("cone")
    coarse_before = _persistent_state(coarse_model)
    coarse = _generate(coarse_model, (coarse_face,), 0.6)

    fine_model, fine_face, surface = _model_face("cone")
    fine_before = _persistent_state(fine_model)
    fine = _generate(fine_model, (fine_face,), 0.3)

    for result, face in ((coarse, coarse_face), (fine, fine_face)):
        mesh = result.mesh
        _assert_linear_shell_topology(mesh, face)
        residual = _support_residual(mesh, surface)
        assert residual <= 1.0e-10, f"support residual {residual} exceeds 1e-10"
        _assert_full_nodes_of_edge_chains(coarse_model if face == coarse_face else fine_model, mesh, face)

        diagnostics = mesh.hybrid_diagnostics
        assert diagnostics["route"] == "quad-first-conical"
        assert diagnostics["geometry_family_by_face"][face] == "conical"
        report = diagnostics["front"]["faces"][face]
        assert report["attempts"] >= report["direct_accepts"] + report["guided_accepts"]

        model = coarse_model if face == coarse_face else fine_model
        domain = ConicalQuadDomain.from_geometry(model, face)
        tiled = sum(
            _element_chart_area(
                domain,
                mesh,
                mesh.quads[element_id] if element_id in mesh.quads else mesh.tris[element_id],
            )
            for element_id in mesh.elements_of_face[face]
        )
        assert tiled > 0.0
        boundary_area = _published_boundary_chart_area(domain, model, mesh, face)
        assert abs(tiled - boundary_area) <= 1.0e-9 * max(1.0, boundary_area)

    assert len(fine.mesh.nodes) > len(coarse.mesh.nodes)
    assert _n_eq(fine.mesh) >= 2.0 * _n_eq(coarse.mesh)
    assert _persistent_state(coarse_model) == coarse_before
    assert _persistent_state(fine_model) == fine_before


def test_ch8_conical_linear_repeat_is_deterministic_and_source_untouched() -> None:
    first_model, first_face, _ = _model_face("cone")
    first_before = _persistent_state(first_model)
    first = _generate(first_model, (first_face,), 0.4)

    repeat_model, repeat_face, _ = _model_face("cone")
    repeat_before = _persistent_state(repeat_model)
    repeat = _generate(repeat_model, (repeat_face,), 0.4)

    assert (len(repeat.mesh.nodes), len(repeat.mesh.quads), len(repeat.mesh.tris)) == (
        len(first.mesh.nodes), len(first.mesh.quads), len(first.mesh.tris)
    )
    first_positions = {tuple(np.asarray(point, dtype=float)) for point in first.mesh.nodes.values()}
    repeat_positions = {tuple(np.asarray(point, dtype=float)) for point in repeat.mesh.nodes.values()}
    assert first_positions == repeat_positions
    assert _persistent_state(first_model) == first_before
    assert _persistent_state(repeat_model) == repeat_before


def test_ch8_conical_cancellation_before_publication_is_atomic() -> None:
    model, face, _ = _model_face("cone")
    before = _persistent_state(model)

    class Cancelled(RuntimeError):
        pass

    def cancel(phase: str) -> None:
        if phase in ("quad-first:face-seed", "quad-first:before-publication"):
            raise Cancelled(phase)

    with pytest.raises(Cancelled):
        _generate(model, (face,), 0.4, cancellation_check=cancel)
    assert _persistent_state(model) == before


def test_ch8_linear_conical_route_survives_later_quadratic_activation() -> None:
    model, face, _ = _model_face("cone")
    before = _persistent_state(model)
    result = _generate(model, (face,), 0.4, order="linear")
    assert result.mesh.order == "linear"
    assert result.mesh.hybrid_diagnostics["route"] == "quad-first-conical"
    assert result.mesh.hybrid_diagnostics["geometry_family_by_face"][face] == "conical"
    assert _persistent_state(model) == before


def test_ch8_cylinder_route_is_untouched_by_conical_activation() -> None:
    model, selected = _sector_model(False)
    face = int(model.face_uses[selected[0].id].face_id)
    surface = model.faces[face].surface
    before = _persistent_state(model)
    result = _generate(model, (face,), 0.5)
    mesh = result.mesh

    assert mesh.quads
    assert set(mesh.elements_of_face) == {face}
    assert mesh.hybrid_diagnostics["route"] == "quad-first-cylindrical"
    assert mesh.hybrid_diagnostics["geometry_family_by_face"][face] == "cylindrical"
    assert _support_residual(mesh, surface) <= 1.0e-10
    assert _persistent_state(model) == before


def test_ch8_planar_route_is_not_hijacked_by_conical_activation() -> None:
    geometry = GeometryModel()
    points = [
        geometry.add_point(*p)
        for p in ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    ]
    edges = [geometry.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    face = geometry.add_face(edges, corners=(0, 1, 2, 3))
    before = _persistent_state(geometry)

    result = _generate(geometry, (face,), 0.4)
    mesh = result.mesh

    assert mesh.quads
    set_of_faces = set(mesh.elements_of_face)
    assert set_of_faces == {face}
    assert mesh.hybrid_diagnostics["route"] != "quad-first-conical"
    assert mesh.hybrid_diagnostics["geometry_family_by_face"][face] != "conical"
    assert _persistent_state(geometry) == before
