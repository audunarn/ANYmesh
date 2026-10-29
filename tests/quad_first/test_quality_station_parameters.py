"""Quality repair consumes validated station parameters without reinversion."""
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.nurbs import NurbsCurve
from anymesher.hybrid import _quality_boundary_midpoints
from anymesher.errors import MeshError


def fixture():
    geometry = GeometryModel()
    curve = NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
        ((0., 0., 0.), (0., 1., 2.), (0., 3., 0.)))
    vertices = [geometry.add_point(*point) for point in curve.evaluate([0., 1.])]
    edge = geometry.add_curve(*vertices, curve)
    params = (0., .2, .7, 1.)
    nodes = {index: point for index, point in enumerate(geometry.sample_edge(edge, params))}
    mesh = SimpleNamespace(nodes=nodes, nodes_of_edge={edge: list(nodes)})
    known = {edge: dict(enumerate(params))}
    return geometry, mesh, edge, params, known


def test_exact_station_parameters_avoid_closest_queries_without_mutation(monkeypatch):
    geometry, mesh, edge, params, known = fixture()
    original = deepcopy(mesh.nodes)
    revision = geometry.revision
    monkeypatch.setattr(geometry, 'closest_edge_point', lambda *args: pytest.fail('known station reinverted'))
    result = _quality_boundary_midpoints(mesh, geometry, known)
    for index, (a, b) in enumerate(zip(params, params[1:])):
        np.testing.assert_array_equal(result[index, index+1], geometry.sample_edge(edge, [(a+b)/2])[0])
    assert geometry.revision == revision
    for node in mesh.nodes:
        np.testing.assert_array_equal(mesh.nodes[node], original[node])


@pytest.mark.parametrize('invalid', ['moved', 'nan', 'order', 'foreign'])
def test_invalid_or_moved_provenance_is_refused(monkeypatch, invalid):
    geometry, mesh, edge, params, known = fixture()
    if invalid == 'moved':
        mesh.nodes[1] = mesh.nodes[1] + (0., .01, 0.)
    elif invalid == 'nan':
        known[edge][1] = float('nan')
    elif invalid == 'order':
        known[edge][1], known[edge][2] = known[edge][2], known[edge][1]
    else:
        known[edge][99] = .5
    before = deepcopy(mesh.nodes)
    monkeypatch.setattr(geometry, 'closest_edge_point', lambda *args: pytest.fail('invalid provenance reinverted'))
    with pytest.raises(MeshError, match='station parameter'):
        _quality_boundary_midpoints(mesh, geometry, known)
    for node in mesh.nodes:
        np.testing.assert_array_equal(mesh.nodes[node], before[node])


def test_unknown_parameters_keep_original_closest_point_fallback(monkeypatch):
    geometry, mesh, edge, params, known = fixture()
    calls = []
    def closest(edge_id, point):
        node = next(node for node, value in mesh.nodes.items() if np.array_equal(value, point))
        calls.append(node)
        return np.array(point), params[node], 0.
    monkeypatch.setattr(geometry, 'closest_edge_point', closest)
    legacy = _quality_boundary_midpoints(mesh, geometry)
    assert calls == [0, 1, 1, 2, 2, 3]
    calls.clear()
    partial = _quality_boundary_midpoints(mesh, geometry, {edge: {0: 0., 3: 1.}})
    assert calls == [1, 1, 2, 2]
    for key in legacy:
        np.testing.assert_array_equal(partial[key], legacy[key])
