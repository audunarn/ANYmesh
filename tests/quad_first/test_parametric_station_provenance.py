"""Known source stations retain exact owner boundary parameters."""
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.nurbs import NurbsCurve
from anymesher.errors import MeshError
from anymesher.quad.domain import ParametricQuadDomain
from anymesher.quad.boundary import BoundaryStationRegistry


def fixture(*, vector=(5., 0., 0.), controls=None):
    geometry = GeometryModel()
    curve = NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
        controls or ((0., 0., 0.), (0., 1.3, 2.1), (0., 3.7, .2)))
    start, end = (geometry.add_point(*point) for point in curve.evaluate([0., 1.]))
    edge = geometry.add_curve(start, end, curve)
    face, = geometry.extrude((edge,), vector)
    domain = ParametricQuadDomain.from_geometry(geometry, face)
    registry = BoundaryStationRegistry.for_domain(geometry, domain, .25)
    return geometry, domain, registry


def test_all_boundary_station_lifts_reproduce_owner_points():
    geometry, domain, registry = fixture()
    revision = geometry.revision
    stations = registry.outer_stations(domain)
    assert any(station.key.kind == 'edge' for station in stations)
    assert any(station.key.kind == 'vertex' for station in stations)
    for station in stations:
        chart = domain.project_station(station)
        np.testing.assert_allclose(domain.lift(chart), station.position, atol=1e-12, rtol=0.)
    assert geometry.revision == revision
    geometry.add_point(99., 99., 99.)
    with pytest.raises(MeshError, match='changed'):
        domain.project_station(stations[0])


def test_wrong_station_coordinates_are_not_silently_snapped():
    geometry, domain, registry = fixture()
    station = next(item for item in registry.outer_stations(domain) if item.key.kind == 'edge')
    wrong = replace(station, position=tuple(np.asarray(station.position) + (0., .01, 0.)))
    with pytest.raises(MeshError, match='inconsistent owner chart'):
        domain.project_station(wrong)


def test_nonouter_station_preserves_legacy_projection(monkeypatch):
    geometry, domain, registry = fixture()
    station = next(item for item in registry.outer_stations(domain) if item.key.kind == 'edge')
    # A hole edge has authenticated station provenance but is outside this
    # optional outer-boundary mapping capability.
    foreign = replace(station, key=replace(station.key, entity_id=9999))
    monkeypatch.setattr(ParametricQuadDomain, 'project', lambda self, point: (12., 34.))
    assert domain.project_station(foreign) == (12., 34.)


def test_seed_preserves_known_station_lifts():
    from anymesher.quad.seed import build_planar_quad_seed
    geometry, domain, registry = fixture()
    seed = build_planar_quad_seed(geometry, domain.face_id, .25,
                                 domain=domain, registry=registry, layout_policy='adaptive')
    assert seed.triangulation.actual_backend == 'structured-rectangle'
    stations = {station.key: station for station in registry.outer_stations(domain)}
    for key, local in seed.station_to_node.items():
        np.testing.assert_allclose(domain.lift(seed.state.position(local)),
                                   stations[key].position, atol=1e-12, rtol=0.)
    interior = set(seed.state.nodes) - set(seed.station_to_node.values())
    assert interior
    for local in interior:
        point = seed.state.position(local)
        uv = np.asarray(domain.inverse_transform) @ point
        np.testing.assert_allclose(domain.lift(point), geometry.face_point(domain.face_id, *uv),
                                   atol=1e-12, rtol=0.)


def test_oblique_or_unsupported_parametric_charts_decline_tensor(monkeypatch):
    from anymesher.quad.rectangle_seed import rectangle_seed_data
    controls = ((0., 0., 0.), (0., 1., 2.), (0., 4., 0.))
    geometry, domain, registry = fixture(vector=(5., 2., 0.), controls=controls)
    outer = tuple(domain.project_station(station) for station in registry.outer_stations(domain))
    assert rectangle_seed_data(geometry, domain, registry, outer, None, None, target_size=.25) is None
    monkeypatch.setattr(GeometryModel, 'face_boundary_uv', lambda *args: None)
    assert rectangle_seed_data(geometry, domain, registry, outer, None, None, target_size=.25) is None
