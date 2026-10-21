"""Nominal planar native size without mapped-axis propagation."""
import numpy as np
import pytest

from anygeometry import GeometryModel, Plane
from anygeometry.entities import OrientedEdge
from anymesher import generate_hybrid_mesh
from anymesher.refinement import refine_at
import anymesher.hybrid as hybrid


def rectangle(width=4., height=2.):
    model = GeometryModel()
    vertices = model.add_points(((0., 0., 0.), (width, 0., 0.),
                                 (width, height, 0.), (0., height, 0.)))
    face = model.add_plate(vertices)
    return model, face


class Captured(BaseException):
    pass


def capture_surface(monkeypatch, model, **kwargs):
    captured = {}
    def stop(outer, holes, **options):
        captured.update(options)
        captured['outer'] = np.asarray(outer).copy()
        raise Captured()
    monkeypatch.setattr(hybrid, 'mesh_planar_surface', stop)
    with pytest.raises(Captured):
        generate_hybrid_mesh(model, strategy='native', order='quadratic',
                             native_backend='python', **kwargs)
    return captured


def test_planar_native_pins_do_not_pin_the_opposite_edge_or_interior(monkeypatch):
    model, face = rectangle()
    bottom, right, top, left = [item.edge for item in model.faces[face].loop]
    captured = capture_surface(monkeypatch, model, target_size=1.,
                               overrides={bottom: 16, top: 8, right: 8, left: 8})
    # All registered segments are <= .5, but the uniform interior remains 1 m.
    assert captured['target_size'] == 1.
    assert len(captured['outer']) == 40


def test_nonuniform_field_keeps_boundary_derived_interior_step(monkeypatch):
    model, face = rectangle()
    pins = {item.edge: 16 for item in model.faces[face].loop}
    captured = capture_surface(monkeypatch, model, target_size=1., overrides=pins,
                               refinements=[refine_at((0., 0., 0.), .25)])
    steps = np.linalg.norm(np.roll(captured['outer'], -1, axis=0)-captured['outer'], axis=1)
    assert captured['target_size'] == max(steps)*(1.+64.*np.finfo(float).eps)
    assert captured['target_size'] < 1.


def test_unpinned_nonuniform_thin_face_keeps_original_seed_constraints(monkeypatch):
    model, face = rectangle(4., .125)
    original = hybrid.solve_seeding
    captured = {}
    def stop(geometry, **kwargs):
        captured['unstructured'] = tuple(kwargs['unstructured_face_ids'])
        actual = original(geometry, **kwargs)
        reference = original(geometry, **dict(kwargs, unstructured_face_ids=()))
        assert actual.divisions == reference.divisions
        captured['divisions'] = actual.divisions
        raise Captured()
    monkeypatch.setattr(hybrid, 'solve_seeding', stop)
    with pytest.raises(Captured):
        generate_hybrid_mesh(model, target_size=1., strategy='native', order='quadratic',
            refinements=[refine_at((0., 0., 0.), .25)], native_backend='python')
    assert captured['unstructured'] == ()
    long_edges = [item.edge for item in model.faces[face].loop
                  if model.edge_length(item.edge) == 4.]
    assert all(captured['divisions'][edge] > 4 for edge in long_edges)


@pytest.mark.parametrize('size', [.5, 1.])
def test_native_public_quadratic_size_and_validity(size):
    model, face = rectangle()
    mesh = generate_hybrid_mesh(model, target_size=size, strategy='native',
                                order='quadratic', native_backend='python')
    for item in model.faces[face].loop:
        expected = round(model.edge_length(item.edge)/size)
        assert mesh.seeding.divisions[item.edge] == expected
        assert len(mesh.nodes_of_edge[item.edge]) == 2*expected+1
    assert mesh.shells
    for element in mesh.shells:
        body = mesh.quads.get(element, mesh.tris.get(element))
        assert len(body) in (6, 8)
        xyz = np.asarray([mesh.nodes[n] for n in mesh.corners_of(element)])
        assert np.sum(xyz[:, 0]*np.roll(xyz[:, 1], -1)-xyz[:, 1]*np.roll(xyz[:, 0], -1)) > 0.


def test_public_native_half_size_increases_shell_count():
    model, _ = rectangle()
    counts = [len(generate_hybrid_mesh(model, target_size=size, strategy='native',
                    order='quadratic', native_backend='python').shells) for size in (1., .5)]
    assert counts[1] > counts[0]


def test_mapped_neighbour_retains_equality_and_quadratic_shared_nodes():
    model = GeometryModel()
    a, b, c, d, e = model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0),
                                     (0, 1, 0), (2, .5, 0)))
    bottom, shared, top, opposite, lower, upper = [model.add_line(*pair)
        for pair in ((a, b), (b, c), (c, d), (d, a), (b, e), (e, c))]
    mapped = model.add_face_from_loop(tuple(OrientedEdge(edge, True)
        for edge in (bottom, shared, top, opposite)), corners=(0, 1, 2, 3))
    native = model.add_face_from_loop((OrientedEdge(shared, False),
        OrientedEdge(lower, True), OrientedEdge(upper, True)),
        surface=Plane(np.array((1., 0., 0.)), np.array((1., 0., 0.)), np.array((0., 1., 0.))))
    mesh = generate_hybrid_mesh(model, target_size=.5, strategy='auto', order='quadratic',
                                overrides={opposite: 4}, native_backend='python')
    assert mesh.seeding.divisions[shared] == mesh.seeding.divisions[opposite] == 4
    shared_nodes = set(mesh.nodes_of_edge[shared])
    assert len(shared_nodes) == 9
    for face in (mapped, native):
        used = {node for element in mesh.elements_of_face[face]
                for node in mesh.quads.get(element, mesh.tris.get(element))}
        assert shared_nodes <= used


def test_nonaffine_coons_keeps_existing_seeding_admission(monkeypatch):
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (3, 0, 0), (2, 1, 0), (0, 1, 0)))
    face = model.add_face(model.add_polyline(points, close=True),
                          corners=(0, 1, 2, 3), surface=None)
    assert hybrid._native_planar_support(model, face) is None
    def stop(geometry, **kwargs):
        assert tuple(kwargs['unstructured_face_ids']) == ()
        raise Captured()
    monkeypatch.setattr(hybrid, 'solve_seeding', stop)
    with pytest.raises(Captured):
        generate_hybrid_mesh(model, target_size=1., strategy='native', order='quadratic')


def test_cancel_exception_survives_planar_seeding():
    model, _ = rectangle()
    marker = RuntimeError('owned cancellation')
    def cancel(stage):
        if stage == 'global seed compatibility and growth':
            raise marker
    with pytest.raises(RuntimeError) as error:
        generate_hybrid_mesh(model, target_size=1., strategy='native', order='quadratic',
                             cancellation_check=cancel)
    assert error.value is marker
