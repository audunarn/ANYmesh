from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from anygeometry import GeometryModel, punch_hole
from anygeometry.entities import OrientedEdge
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.boundary import BoundaryStationRegistry
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.rectangle_seed import rectangle_seed_data
from anymesher.quad.seed import build_planar_quad_seed
from anymesher.refinement import SizeField


def rectangle(*, reversed=False, rotated=False, skew=0., width=1.2):
    model = GeometryModel()
    points = np.asarray(((0., 0., 0.), (3., 0., 0.),
                         (3.+skew, width, 0.), (skew, width, 0.)))
    if rotated:
        # Exact orthogonal axes in a general 3-D plane, plus translation.
        points = points @ np.asarray(((.6, .8, 0.), (-.48, .36, .8), (.64, -.48, .6)))
        points += (13., -4., 7.)
    if reversed:
        points = points[::-1]
    vertices = model.add_points(points)
    face = model.add_face(model.add_polyline(vertices, close=True), surface=None)
    return model, face


def capture(model, face, h=.5):
    domain = PlanarQuadDomain.from_geometry(model, face)
    registry = BoundaryStationRegistry.for_domains(model, (domain,), h,
                                                   _adaptive_independent_counts=True)
    outer = tuple(domain.project(s.position) for s in registry.outer_stations(domain))
    return domain, registry, outer


@pytest.mark.parametrize('rotated,reversed', ((False, False), (True, False), (True, True)))
def test_tensor_seed_preserves_every_station_and_area(rotated, reversed):
    model, face = rectangle(rotated=rotated, reversed=reversed)
    revision = model.revision
    domain, registry, outer = capture(model, face)
    seed = build_planar_quad_seed(model, face, .5, domain=domain, registry=registry,
                                  layout_policy='adaptive')
    assert seed.triangulation.actual_backend == 'structured-rectangle'
    assert set(seed.state.cell_kinds.values()) == {'Q4'}
    assert len(seed.state.cells) == 12
    assert not seed.state.front
    assert seed.discrete_area == pytest.approx(3.6, abs=1e-12)
    assert len(seed.triangulation.triangles) == 2*len(seed.state.cells)
    np.testing.assert_array_equal(seed.triangulation.points[:len(outer)], outer)
    assert seed.state.protected_nodes == frozenset(seed.station_to_node.values())
    assert tuple(seed.station_to_node) == tuple(s.key for s in registry.outer_stations(domain))
    assert model.revision == revision


@pytest.mark.parametrize('order', ('linear', 'quadratic'))
def test_public_regular_bay_size_response_and_validity(order):
    model, face = rectangle(rotated=True)
    args = dict(face_ids=(face,), layout_policy='adaptive', order=order,
                quad_options=QuadMeshingOptions(quality_model='shape_jacobian'))
    coarse = generate_hybrid_mesh_result(model, target_size=1., **args).mesh
    fine = generate_hybrid_mesh_result(model, target_size=.5, **args).mesh
    assert len(fine.quads) > len(coarse.quads)
    assert not fine.tris and not coarse.tris
    for body in fine.quads.values():
        xyz = np.asarray([fine.nodes[n] for n in body[:4]])
        edges = np.roll(xyz, -1, axis=0)-xyz
        cosine = np.sum(edges*np.roll(edges, 1, axis=0), axis=1)
        np.testing.assert_allclose(cosine, 0., atol=1e-12, rtol=0.)
    if order == 'quadratic':
        assert fine.hybrid_diagnostics['high_order_geometry']['status'] == 'CERTIFIED_POSITIVE'


@pytest.mark.parametrize('failure', ('counts', 'distribution', 'skew', 'hole', 'field', 'domain'))
def test_unsupported_cases_decline_without_resampling(failure):
    model, face = rectangle(skew=.25 if failure == 'skew' else 0.)
    if failure == 'hole':
        face, _ = punch_hole(model, face, (1.5, .625, 0.), .15)
    domain, registry, outer = capture(model, face)
    field = None
    if failure in ('counts', 'distribution'):
        edge = domain.edge_uses[2][0]
        chains = dict(registry.chains)
        chain = list(chains[edge])
        if failure == 'counts':
            del chain[1]
        else:
            point = tuple(.75*a+.25*b for a, b in zip(chain[1].position, chain[2].position))
            chain[1] = replace(chain[1], position=point)
        chains[edge] = tuple(chain)
        registry = replace(registry, chains=MappingProxyType(chains))
    if failure == 'field':
        field = SizeField(model, .5)
        field.zones = (object(),)
    if failure == 'domain':
        class CustomDomain(PlanarQuadDomain):
            pass
        domain = CustomDomain(**domain.__dict__)
    assert rectangle_seed_data(model, domain, registry, outer, field, None, target_size=.5) is None


def test_existing_policy_and_attachment_cut_keep_original_route():
    model, face = rectangle()
    seed = build_planar_quad_seed(model, face, .5)
    assert seed.triangulation.actual_backend != 'structured-rectangle'
    seed = build_planar_quad_seed(model, face, .5, layout_policy='adaptive',
                                  source_aligned_cut=(.5, .5))
    assert seed.triangulation.actual_backend != 'structured-rectangle'


def test_transformed_rounding_threshold_keeps_unequal_canonical_counts():
    model, face = rectangle(rotated=True, width=1.25)
    domain, registry, outer = capture(model, face)
    counts = tuple(len(registry.chain(edge, forward)) for edge, forward in domain.edge_uses)
    assert counts == (7, 4, 7, 3)
    assert rectangle_seed_data(model, domain, registry, outer, None, None, target_size=.5) is None
    seed = build_planar_quad_seed(model, face, .5, domain=domain, registry=registry,
                                  layout_policy='adaptive')
    assert seed.triangulation.actual_backend != 'structured-rectangle'
    np.testing.assert_array_equal(seed.triangulation.points[:len(outer)], outer)


def test_cancellation_and_revision_callbacks_do_not_publish():
    model, face = rectangle()
    before = model.revision
    marker = RuntimeError('cancel structured seed')
    def cancel(stage):
        if stage == 'quad-first:structured-rectangle':
            raise marker
    with pytest.raises(RuntimeError) as caught:
        build_planar_quad_seed(model, face, .5, layout_policy='adaptive', cancellation_check=cancel)
    assert caught.value is marker
    assert model.revision == before
    def mutate(stage):
        if stage == 'quad-first:structured-rectangle':
            model.add_point(9., 9., 9.)
    from anymesher.errors import MeshError
    with pytest.raises(MeshError, match='changed'):
        build_planar_quad_seed(model, face, .5, layout_policy='adaptive', cancellation_check=mutate)


def test_shared_boundary_and_quadratic_midnodes_are_conforming():
    model = GeometryModel()
    v = model.add_points(((0,0,0), (3,0,0), (3,1.25,0), (0,1.25,0), (6,0,0), (6,1.25,0)))
    a = model.add_line(v[0], v[1]); shared = model.add_line(v[1], v[2])
    b = model.add_line(v[2], v[3]); c = model.add_line(v[3], v[0])
    d = model.add_line(v[1], v[4]); e = model.add_line(v[4], v[5]); f = model.add_line(v[5], v[2])
    f1 = model.add_face_from_loop(tuple(OrientedEdge(edge, True) for edge in (a, shared, b, c)))
    f2 = model.add_face_from_loop((OrientedEdge(d,True), OrientedEdge(e,True),
                                   OrientedEdge(f,True), OrientedEdge(shared,False)))
    mesh = generate_hybrid_mesh_result(model, target_size=.5, face_ids=(f1,f2),
        order='quadratic', layout_policy='adaptive', quad_options=QuadMeshingOptions()).mesh
    left = {n for eid in mesh.elements_of_face[f1] for n in mesh.quads[eid]}
    right = {n for eid in mesh.elements_of_face[f2] for n in mesh.quads[eid]}
    assert len(left & right) == 5  # Two spans: three corners plus two Q8 mids.
    assert not mesh.tris


def test_fine_candidate_budget_refusal_matches_original_before_grid_allocation(monkeypatch):
    from anymesher.errors import MeshError
    from anymesher.quad.seed import _adaptive_interior_points
    model, face = rectangle()
    domain = PlanarQuadDomain.from_geometry(model, face)
    registry = BoundaryStationRegistry.for_domains(model, (domain,), .001,
        overrides={edge: 1 for edge, _ in domain.edge_uses})
    outer = tuple(domain.project(s.position) for s in registry.outer_stations(domain))
    with pytest.raises(MeshError, match='fine candidate budget'):
        _adaptive_interior_points(domain, outer, (), .001, None, None)
    original = np.empty
    def guarded(shape, *args, **kwargs):
        assert not isinstance(shape, tuple) or np.prod(shape) < 250000
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'empty', guarded)
    with pytest.raises(MeshError, match='fine candidate budget'):
        build_planar_quad_seed(model, face, .001, domain=domain, registry=registry,
                               layout_policy='adaptive')


def test_dense_station_product_declines_before_allocating_grid(monkeypatch):
    model, face = rectangle()
    domain = PlanarQuadDomain.from_geometry(model, face)
    registry = BoundaryStationRegistry.for_domains(model, (domain,), 1.,
        overrides={edge: 500 for edge, _ in domain.edge_uses})
    outer = tuple(domain.project(s.position) for s in registry.outer_stations(domain))
    original = np.empty
    def guarded(shape, *args, **kwargs):
        assert not isinstance(shape, tuple) or np.prod(shape) <= 250000
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'empty', guarded)
    assert rectangle_seed_data(model, domain, registry, outer, None, None, target_size=1.) is None


def test_single_row_cell_construction_remains_cancellable():
    model, face = rectangle()
    marker = RuntimeError('cancel cell construction')
    def cancel(stage):
        if stage == 'quad-first:structured-rectangle-cells':
            raise marker
    with pytest.raises(RuntimeError) as caught:
        build_planar_quad_seed(model, face, 1., layout_policy='adaptive', cancellation_check=cancel)
    assert caught.value is marker


def test_budget_uses_requested_size_not_compatible_registry_roundoff(monkeypatch):
    model = GeometryModel()
    vertices = model.add_points(((0.,0.,0.), (250.,0.,0.), (250.,250.,0.), (0.,250.,0.)))
    face = model.add_face(model.add_polyline(vertices, close=True), surface=None)
    domain = PlanarQuadDomain.from_geometry(model, face)
    registry = BoundaryStationRegistry.for_domains(model, (domain,), np.nextafter(1., 0.),
        overrides={edge: 1 for edge, _ in domain.edge_uses})
    # Original adaptive budget uses requested h=1: exactly500*500, allowed.
    # Registry h is accepted by existing compatibility checks, but using it
    # instead would ceil to501*501 and incorrectly refuse this request.
    assert int(np.ceil(250./(1./2.)))**2 == 250000
    assert int(np.ceil(250./(registry.target_size/2.)))**2 > 250000
    marker = RuntimeError('reached small tensor allocation')
    original = np.empty
    def stop_at_grid(shape, *args, **kwargs):
        if type(shape) is tuple and shape == (2, 2):
            raise marker
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'empty', stop_at_grid)
    with pytest.raises(RuntimeError) as caught:
        build_planar_quad_seed(model, face, 1., domain=domain, registry=registry,
                               layout_policy='adaptive')
    assert caught.value is marker
