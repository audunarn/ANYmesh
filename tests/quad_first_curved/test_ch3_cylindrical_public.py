from __future__ import annotations

import math

import anygeometry
import numpy as np
import pytest

from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state

from anymesher._cylindrical_patch import prepare_cylindrical_patch
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported


def _face_id(model, face_use) -> int:
    return int(model.face_uses[face_use.id].face_id)


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


def _face_nodes(mesh, face_id: int) -> set[int]:
    nodes: set[int] = set()
    for element_id in mesh.elements_of_face[int(face_id)]:
        if element_id in mesh.quads:
            nodes.update(mesh.quads[element_id])
        elif element_id in mesh.tris:
            nodes.update(mesh.tris[element_id])
        else:
            raise AssertionError(f"face {face_id} owns unknown shell {element_id}")
    return nodes


def _assert_on_unit_cylinder(mesh, *, theta_max: float | None = None) -> None:
    points = np.asarray(tuple(mesh.nodes.values()), dtype=float)
    assert points.ndim == 2 and points.shape[1] == 3 and len(points)
    np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 1.0, atol=1.0e-10)
    assert np.all(points[:, 2] >= -1.0e-10)
    assert np.all(points[:, 2] <= 2.0 + 1.0e-10)
    if theta_max is not None:
        theta = np.arctan2(points[:, 1], points[:, 0])
        assert np.all(theta >= -1.0e-10)
        assert np.all(theta <= float(theta_max) + 1.0e-10)


def _n_eq(mesh) -> float:
    return float(len(mesh.quads)) + 0.5 * float(len(mesh.tris))


@pytest.mark.parametrize(
    "sector,target_size,origin,rotation",
    ((0, .38, (0., 0., 0.), 0.),
     (3, .38, (0., 0., 0.), 0.),
     # pi/4 is only 0.025 m beyond two 0.38 m steps. This used to
     # create the narrow remainder strip seen in RA1.
     (0, .38, (2., -3., 4.), math.pi/6),
     (3, .38, (2., -3., 4.), math.pi/6)),
)
def test_cylindrical_seed_follows_shared_station_axes(
    sector: int, target_size: float, origin, rotation: float,
) -> None:
    from anymesher.quad.boundary import BoundaryStationRegistry
    from anymesher.quad.domain import CylindricalQuadDomain
    from anymesher.quad.seed import build_planar_quad_seed

    model, selected = _sector_model(False, origin=origin, rotation=rotation)
    face = _face_id(model, selected[sector])
    binding = prepare_cylindrical_patch(model, (selected[sector],))
    domain = CylindricalQuadDomain.from_binding(model, face, binding)
    registry = BoundaryStationRegistry.for_domain(model, domain, target_size)
    seed = build_planar_quad_seed(model, face, target_size, domain=domain, registry=registry)
    arc = next(edge_id for edge_id, _ in domain.edge_uses
               if type(model.edges[edge_id].curve).__name__ == "Arc")
    assert registry.edge_divisions(arc) == 3
    boundary = registry.outer_stations(domain)
    chart = np.asarray([domain.project(station.position) for station in boundary])
    interior = np.asarray([seed.state.nodes[node] for node in seed.state.nodes
                           if node not in seed.state.protected_nodes])
    assert len(interior)
    for axis in (0, 1):
        edge_axis = np.unique(np.round(chart[:, axis], 10))
        assert all(np.min(abs(edge_axis-value)) <= 1e-9
                   for value in np.unique(np.round(interior[:, axis], 10)))


def test_quad_stage_timings_are_a_sidecar() -> None:
    from anymesher.quad.timing import collect_quad_stage_timings

    model, selected = _sector_model(False)
    with collect_quad_stage_timings() as timings:
        mesh = generate_hybrid_mesh_result(
            model, face_ids=(_face_id(model, selected[0]),), target_size=0.5,
            strategy="native", native_backend="python", order="quadratic",
            quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
        ).mesh
    assert timings["qualification"] > 0.0
    assert timings["seeding"] > 0.0
    assert timings["repair"] > 0.0
    assert timings["strict_validation"] > 0.0
    assert "timings" not in mesh.hybrid_diagnostics


def test_ch3_cylindrical_domain_from_owner_patch() -> None:
    model, selected = _sector_model(False)
    before = _persistent_state(model)
    face = _face_id(model, selected[0])
    binding = prepare_cylindrical_patch(model, (selected[0],))

    from anymesher.quad.domain import CylindricalQuadDomain

    domain = CylindricalQuadDomain.from_binding(model, face, binding)
    assert domain.model_id == str(model.model_id)
    assert domain.revision == int(model.revision)
    assert domain.face_id == face
    assert domain.area == pytest.approx(math.pi / 2.0, abs=1.0e-12)
    for vertex_id in domain.vertex_ids:
        world = model.vertex_position(vertex_id)
        assert np.allclose(domain.lift(domain.project(world)), world, atol=1.0e-11)
    assert _persistent_state(model) == before


def test_single_sector_linear_quads_refine_with_target_size() -> None:
    coarse_model, coarse_selected = _sector_model(False)
    coarse_before = _persistent_state(coarse_model)
    coarse_face = _face_id(coarse_model, coarse_selected[0])
    coarse = _generate(coarse_model, (coarse_face,), 0.5)

    fine_model, fine_selected = _sector_model(False)
    fine_before = _persistent_state(fine_model)
    fine_face = _face_id(fine_model, fine_selected[0])
    fine = _generate(fine_model, (fine_face,), 0.25)

    for result, face in ((coarse, coarse_face), (fine, fine_face)):
        mesh = result.mesh
        assert mesh.quads
        assert set(mesh.elements_of_face) == {face}
        assert set(mesh.elements_of_face[face]) == set(mesh.quads) | set(mesh.tris)
        _assert_on_unit_cylinder(mesh, theta_max=np.pi / 4.0)
        source_model = coarse_model if face == coarse_face else fine_model
        for use in source_model.faces[face].loop:
            assert int(use.edge) in mesh.nodes_of_edge
            assert len(mesh.nodes_of_edge[int(use.edge)]) >= 2
        assert mesh.hybrid_diagnostics["route"] == "quad-first-cylindrical"
        assert mesh.hybrid_diagnostics["geometry_family_by_face"][face] == "cylindrical"
        report = mesh.hybrid_diagnostics["front"]["faces"][face]
        assert report["attempts"] >= report["direct_accepts"] + report["guided_accepts"]

    assert _n_eq(fine.mesh) >= 2.0 * _n_eq(coarse.mesh)
    assert len(fine.mesh.nodes) > len(coarse.mesh.nodes)
    assert _persistent_state(coarse_model) == coarse_before
    assert _persistent_state(fine_model) == fine_before


def test_cylindrical_quadratic_is_qualified_after_ch4() -> None:
    # CH4 qualified the previously fail-closed cylindrical quadratic scope;
    # the full contract is asserted in tests/quad_first_curved/test_ch4_cylindrical_quadratic.py.
    model, selected = _sector_model(False)
    before = _persistent_state(model)
    face = _face_id(model, selected[0])
    mesh = _generate(model, (face,), 0.5, order="quadratic").mesh
    assert mesh.order == "quadratic"
    assert mesh.quads
    assert _persistent_state(model) == before


def test_full_ring_periodic_seam_reuses_exact_source_station_ids() -> None:
    model, selected = _sector_model(False)
    before = _persistent_state(model)
    faces = tuple(_face_id(model, face_use) for face_use in selected)
    face_0, face_7 = faces[0], faces[7]
    seam_edges = [
        edge_id for edge_id in model.edges
        if set(model.faces_using_edge(edge_id)) == {face_0, face_7}
    ]
    assert len(seam_edges) == 1
    seam_edge = int(seam_edges[0])

    result = _generate(model, faces, 0.5)
    mesh = result.mesh
    assert set(mesh.elements_of_face) == set(faces)
    _assert_on_unit_cylinder(mesh)
    chain = tuple(mesh.nodes_of_edge[seam_edge])
    assert len(chain) >= 2 and len(chain) == len(set(chain))
    assert set(chain) <= _face_nodes(mesh, face_0)
    assert set(chain) <= _face_nodes(mesh, face_7)
    assert len({tuple(np.asarray(mesh.nodes[n], dtype=float)) for n in chain}) == len(chain)
    assert _persistent_state(model) == before

    repeat_model, repeat_selected = _sector_model(False)
    repeat_before = _persistent_state(repeat_model)
    repeat_faces = tuple(_face_id(repeat_model, face_use) for face_use in repeat_selected)
    repeat = _generate(repeat_model, repeat_faces, 0.5)
    assert (len(repeat.mesh.nodes), len(repeat.mesh.quads), len(repeat.mesh.tris)) == (
        len(mesh.nodes), len(mesh.quads), len(mesh.tris)
    )
    assert _persistent_state(repeat_model) == repeat_before


def test_adjacent_rotated_patches_share_reversed_quadratic_edge() -> None:
    model, selected = _sector_model(False, origin=(2., -3., 4.), rotation=math.pi/6)
    faces = tuple(_face_id(model, use) for use in selected[:2])
    common = [edge for edge in model.edges
              if set(model.faces_using_edge(edge)) == set(faces)]
    assert len(common) == 1
    edge = common[0]
    directions = [next(use.forward for use in model.faces[face].loop
                       if use.edge == edge) for face in faces]
    assert directions == [not directions[1], directions[1]]
    arguments = dict(
        face_ids=faces, target_size=.38, strategy="native",
        native_backend="python", order="quadratic",
        quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
    )
    if not hasattr(anygeometry, "query_cylinder_open_component"):
        # The pinned ANYgeometry predates the owner API for an open two-face
        # cylindrical component; the route must refuse explicitly rather than
        # approximate the missing owner qualification.
        with pytest.raises(QuadPublicUnsupported, match="owner-qualified"):
            generate_hybrid_mesh_result(model, **arguments)
        return
    mesh = generate_hybrid_mesh_result(model, **arguments).mesh
    chain = mesh.nodes_of_edge[edge]
    assert len(chain) >= 5 and len(chain) == len(set(chain))
    assert all(set(chain) <= _face_nodes(mesh, face) for face in faces)
    assert mesh.hybrid_diagnostics["high_order_geometry"]["status"] == "CERTIFIED_POSITIVE"


def test_cylindrical_public_cancellation_has_no_partial_publication() -> None:
    model, selected = _sector_model(False)
    before = _persistent_state(model)
    face = _face_id(model, selected[0])

    class Cancelled(RuntimeError):
        pass

    def cancel(phase: str) -> None:
        if phase == "quad-first:face-seed":
            raise Cancelled(phase)

    with pytest.raises(Cancelled):
        _generate(model, (face,), 0.5, cancellation_check=cancel)
    assert _persistent_state(model) == before
