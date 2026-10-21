"""Optional native station-product candidate; original gates remain decisive."""
import numpy as np
import pytest

from anygeometry import GeometryModel
from anymesher import generate_hybrid_mesh
from anymesher.errors import MeshError
from anymesher.refinement import refine_at
import anymesher.hybrid as hybrid
import anymesher.surface_mesh as surface


def boundary(n=4):
    t = np.arange(n, dtype=float)/n
    return np.vstack((np.column_stack((2*t, np.zeros(n))),
                      np.column_stack((np.full(n, 2.), 2*t)),
                      np.column_stack((2-2*t, np.full(n, 2.))),
                      np.column_stack((np.zeros(n), 2-2*t))))


@pytest.mark.parametrize('angle', [0., .37])
@pytest.mark.parametrize('reverse', [False, True])
def test_station_product_rotates_and_preserves_boundary(angle, reverse):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    ring = boundary() @ rotation.T
    if reverse:
        ring = ring[::-1].copy()
    saved = ring.copy()
    points = surface._cartesian_rectangle_points(ring, 100, None)
    assert points is not None and points.shape == (9, 2)
    assert np.array_equal(ring, saved)
    expected = np.array([(x, y) for x in (.5, 1., 1.5) for y in (.5, 1., 1.5)])
    local = points @ rotation
    assert all(np.min(np.linalg.norm(expected-point, axis=1)) < 1.e-14 for point in local)


@pytest.mark.parametrize('change', ['count', 'distribution', 'skew', 'duplicate', 'budget'])
def test_unsupported_layout_declines(change):
    ring = boundary()
    budget = 100
    if change == 'count':
        ring = np.delete(ring, 1, axis=0)
    elif change == 'distribution':
        ring[1, 0] += .1
    elif change == 'skew':
        ring[:, 0] += .1*ring[:, 1]
    elif change == 'duplicate':
        ring[1] = ring[0]
    else:
        budget = 8
    assert surface._cartesian_rectangle_points(ring, budget, None) is None


def test_row_cancellation_identity():
    marker = MeshError('cancel Cartesian row')
    def cancel(phase):
        raise marker
    with pytest.raises(MeshError) as caught:
        surface._cartesian_rectangle_points(boundary(), 100, cancel)
    assert caught.value is marker


def test_thin_boundary_only_product_is_valid_and_typed():
    # Exact station layout observed at the first real prepared barge bay.
    bottom = np.column_stack((np.arange(0., 4.5, .5), np.zeros(9)))
    top = np.column_stack((np.arange(4., -.5, -.5), np.full(9, .7000000000000002)))
    points = surface._cartesian_rectangle_points(np.vstack((bottom, top)), 100, None)
    assert points is not None and points.shape == (0, 2) and points.dtype == np.float64


@pytest.mark.parametrize('backend', ['python', 'auto'])
def test_public_native_thin_canonical_strip_selects_eight_rectangles(monkeypatch, backend):
    model = GeometryModel()
    vertices = model.add_points(((0., 0., 0.), (4., 0., 0.),
                                 (4., .7000000000000002, 0.), (0., .7000000000000002, 0.)))
    face_id = model.add_plate(vertices)
    edges = [item.edge for item in model.faces[face_id].loop]
    original = hybrid.mesh_planar_surface
    selected = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        selected.append(kwargs['diagnostics']['quality_optimization']['selected_strategy'])
        return result
    monkeypatch.setattr(hybrid, 'mesh_planar_surface', observe)
    mesh = generate_hybrid_mesh(model, strategy='native', target_size=.5, order='linear',
        native_backend=backend, overrides={edges[0]: 8, edges[1]: 1, edges[2]: 8, edges[3]: 1})
    assert selected == ['cartesian_rectangle']
    assert len(mesh.quads) == 8 and not mesh.tris
    assert [len(mesh.nodes_of_edge[edge])-1 for edge in edges] == [8, 1, 8, 1]
    for nodes in mesh.quads.values():
        points = np.asarray([mesh.nodes[node] for node in nodes])
        delta = np.roll(points, -1, axis=0)-points
        assert np.max(np.abs(np.sum(delta*np.roll(delta, -1, axis=0), axis=1))) < 1.e-14


def test_connected_surface_candidate_keeps_boundary_and_gates():
    ring = boundary()
    diagnostics = {}
    mesh = surface.mesh_planar_surface(ring, target_size=.5, backend='python',
        _boundary_is_seeded=True, _cartesian_rectangle_candidate=True,
        diagnostics=diagnostics)
    assert any(path['strategy'] == 'cartesian_rectangle'
               for path in diagnostics['quality_optimization']['candidates'])
    assert all(any(np.array_equal(node[:2], point) for node in mesh.node_coordinates)
               for point in ring)
    if diagnostics['quality_optimization']['selected_strategy'] == 'cartesian_rectangle':
        assert diagnostics['quality_policy']['accepted']
        growth, poor = surface._published_growth(mesh, 1.5)
        assert not poor
    assert diagnostics['actual_backend'] == 'python'


def test_candidate_cancellation_mesherror_is_not_fallback(monkeypatch):
    original = surface._run_quality_path
    marker = MeshError('cancel inside optional triangulation')
    def path(name, *args, **kwargs):
        if name == 'cartesian_rectangle':
            args[6]('only Cartesian cancellation')
        return original(name, *args, **kwargs)
    def cancel(phase):
        if phase == 'only Cartesian cancellation':
            raise marker
    monkeypatch.setattr(surface, '_run_quality_path', path)
    with pytest.raises(MeshError) as caught:
        surface.mesh_planar_surface(boundary(), target_size=.5, backend='python',
            _boundary_is_seeded=True, _cartesian_rectangle_candidate=True,
            cancellation_check=cancel)
    assert caught.value is marker


def test_candidate_refusal_retains_complete_original(monkeypatch):
    ring = boundary()
    expected = surface.mesh_planar_surface(ring, target_size=.5, backend='python',
                                         _boundary_is_seeded=True)
    original = surface._run_quality_path
    def path(name, *args, **kwargs):
        if name == 'cartesian_rectangle':
            raise MeshError('optional candidate rejected')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(surface, '_run_quality_path', path)
    actual = surface.mesh_planar_surface(ring, target_size=.5, backend='python',
        _boundary_is_seeded=True, _cartesian_rectangle_candidate=True)
    for field in ('node_coordinates', 'triangle_connectivity', 'quad_connectivity',
                  'node_ids', 'triangle_ids', 'quad_ids'):
        assert np.array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize('mode', ['uniform', 'quadratic', 'refined', 'parameterized'])
def test_public_owner_admission(monkeypatch, mode):
    model = GeometryModel()
    vertices = model.add_points(((0., 0., 0.), (2., 0., 0.),
                                 (2., 2., 0.), (0., 2., 0.)))
    face_id = model.add_plate(vertices)
    if mode == 'parameterized':
        model.set_face_parameterization(face_id, model.faces[face_id].surface)
    class Captured(BaseException):
        pass
    captured = {}
    def stop(*args, **kwargs):
        captured.update(kwargs)
        raise Captured()
    monkeypatch.setattr(hybrid, 'mesh_planar_surface', stop)
    with pytest.raises(Captured):
        generate_hybrid_mesh(model, strategy='native', target_size=.5,
            order='quadratic' if mode == 'quadratic' else 'linear',
            native_backend='python',
            refinements=[refine_at((0., 0., 0.), .25)] if mode == 'refined' else ())
    assert captured['_cartesian_rectangle_candidate'] is (mode == 'uniform')
