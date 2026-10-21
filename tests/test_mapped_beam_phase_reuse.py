"""Complete mapped-mesh parity against the original per-beam membership scan."""
import json
import importlib.util
import sys

import numpy as np
import pytest
from anygeometry import GeometryModel, Straight, to_dict
from anymesher import mapped
from anymesher.mesh import Coupling
from anymesher.refinement import refine_at
from anymesher.serialize import mesh_to_dict


def legacy_build_beam(mesh, edge_id, next_element):
    """Frozen original _build_beam algorithm, independent of the new index."""
    plating = mesh.nodes_of_edge[edge_id]
    sequence = mesh.offset_nodes_of_edge.get(edge_id) or plating
    step = 2 if mesh.is_quadratic else 1
    element_ids = []
    for start in range(0, len(sequence)-step, step):
        element_id = next_element.next()
        span = sequence[start:start+step+1]
        mesh.beams[element_id] = tuple(int(node) for node in span)
        element_ids.append(element_id)
    mesh.elements_of_edge[edge_id] = element_ids
    shell_nodes = {int(node) for connectivity in (*mesh.quads.values(), *mesh.tris.values())
                   for node in connectivity}
    if sequence is not plating and all(node in shell_nodes for node in plating):
        for beam_node, plate_node in zip(sequence, plating):
            element_id = next_element.next()
            mesh.couplings[element_id] = Coupling.node_to_node(
                beam_node, plate_node, mesh.nodes[beam_node]-mesh.nodes[plate_node])


def assert_exact_mesh(actual, expected):
    assert json.dumps(mesh_to_dict(actual), sort_keys=True) == json.dumps(mesh_to_dict(expected), sort_keys=True)
    assert list(actual.nodes) == list(expected.nodes)
    for node in actual.nodes:
        assert actual.nodes[node].tobytes() == expected.nodes[node].tobytes()
    assert list(actual.quads) == list(expected.quads)
    assert list(actual.beams) == list(expected.beams)
    assert list(actual.couplings) == list(expected.couplings)
    for face, grid in actual.grid_of_face.items():
        np.testing.assert_array_equal(grid, expected.grid_of_face[face])


def plate(reverse=False):
    model = GeometryModel()
    points = [(0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.)]
    if reverse:
        points.reverse()
    edges = model.add_polyline(model.add_points(points), close=True)
    model.add_face(edges)
    return model, edges


@pytest.mark.parametrize('order', ['linear', 'quadratic'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('offset', [0., .125, (0., -0., .125)])
def test_complete_mapped_beams_match_frozen_scalar(monkeypatch, order, reverse, offset):
    model, edges = plate(reverse)
    before = to_dict(model)
    options = dict(target_size=.25, order=order, beam_edges=edges,
                   overrides={edges[0]: 8}, beam_offsets=dict.fromkeys(edges, offset))
    actual = mapped.generate_mesh(model, **options)
    with monkeypatch.context() as patch:
        patch.setattr(mapped, '_build_beam', legacy_build_beam)
        expected = mapped.generate_mesh(model, **options)
    assert_exact_mesh(actual, expected)
    assert to_dict(model) == before


@pytest.mark.parametrize('curved', [False, True])
def test_refined_and_curved_shells_preserve_complete_mesh(monkeypatch, curved):
    if curved:
        model = GeometryModel()
        start, via, end = model.add_points(((2., 0., 0.), (2**.5, 2**.5, 0.), (0., 2., 0.)))
        arc = model.add_arc(start, via, end)
        model.extrude([arc], (0., 0., 2.))
        edges = [edge for edge, record in model.edges.items() if isinstance(record.curve, Straight)]
        refinements = ()
    else:
        model, edges = plate()
        refinements = (refine_at((0., 0., 0.), .125),)
    options = dict(target_size=.4, order='quadratic', beam_edges=edges,
                   refinements=refinements, beam_offsets=dict.fromkeys(edges, (0., 0., .1)))
    actual = mapped.generate_mesh(model, **options)
    with monkeypatch.context() as patch:
        patch.setattr(mapped, '_build_beam', legacy_build_beam)
        expected = mapped.generate_mesh(model, **options)
    assert_exact_mesh(actual, expected)


def test_custom_offset_builder_cannot_reuse_stale_shell_membership(monkeypatch):
    model, edges = plate()
    original = mapped._build_offset_nodes
    calls = []
    def mutate(geometry, mesh, edge_id, *args):
        calls.append(edge_id)
        if len(calls) == 2:
            mesh.quads.clear()
        return original(geometry, mesh, edge_id, *args)
    monkeypatch.setattr(mapped, '_build_offset_nodes', mutate)
    options = dict(target_size=.5, order='quadratic', beam_edges=edges[:2],
                   beam_offsets=dict.fromkeys(edges[:2], (0., 0., .1)))
    actual = mapped.generate_mesh(model, **options)
    calls.clear()
    monkeypatch.setattr(mapped, '_build_beam', legacy_build_beam)
    expected = mapped.generate_mesh(model, **options)
    assert_exact_mesh(actual, expected)


@pytest.mark.parametrize('invalid', [float('nan'), (0., 0.), [[0., 0., .1]]])
def test_offset_refusal_matches_original(monkeypatch, invalid):
    model, edges = plate()
    options = dict(target_size=.5, order='quadratic', beam_edges=edges,
                   beam_offsets={edges[1]: invalid})
    errors = []
    for method in (mapped._build_beam, legacy_build_beam):
        monkeypatch.setattr(mapped, '_build_beam', method)
        with pytest.raises(Exception) as error:
            mapped.generate_mesh(model, **options)
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]


def test_public_native_beam_phase_uses_one_local_lazy_index():
    import copyreg
    copyreg._slotnames(mapped.Mesh)
    model, edges = plate()
    observed = []
    code = mapped._build_beam.__code__
    previous = sys.getprofile()
    def observe(frame, event, value):
        if event == 'return' and frame.f_code is code:
            cache = frame.f_locals['_shell_node_cache']
            observed.append((id(cache), len(cache) if cache is not None else None))
        if previous is not None:
            previous(frame, event, value)
    sys.setprofile(observe)
    try:
        mesh = mapped.generate_mesh(model, target_size=.25, order='quadratic',
            beam_edges=edges, beam_offsets=dict.fromkeys(edges, (0., 0., .1)))
    finally:
        sys.setprofile(previous)
    assert len(observed) == len(edges)
    assert len(set(observed)) == 1
    assert observed[0][1] == 1
    assert not any('shell_node_cache' in key for key in vars(mesh))


def test_custom_offset_conversion_keeps_call_sequence(monkeypatch):
    model, edges = plate()
    calls = []
    class Offset:
        def __array__(self, dtype=None, copy=None):
            calls.append('offset')
            return np.asarray((0., 0., .1), dtype=dtype)
    options = dict(target_size=.5, order='quadratic', beam_edges=edges,
                   beam_offsets=dict.fromkeys(edges, Offset()))
    actual = mapped.generate_mesh(model, **options)
    actual_calls = list(calls)
    calls.clear()
    monkeypatch.setattr(mapped, '_build_beam', legacy_build_beam)
    expected = mapped.generate_mesh(model, **options)
    assert calls == actual_calls
    assert_exact_mesh(actual, expected)


def test_preimport_replaced_mesh_class_does_not_become_native_anchor(monkeypatch):
    from anymesher import mesh as mesh_module
    class CustomMesh(mesh_module.Mesh):
        pass
    monkeypatch.setattr(mesh_module, 'Mesh', CustomMesh)
    spec = importlib.util.spec_from_file_location('anymesher._mapped_custom_import_test', mapped.__file__)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.Mesh is CustomMesh
    assert not loaded._native_beam_phase(GeometryModel(), CustomMesh())


@pytest.mark.parametrize('descriptor,preimport', [('shells', False), ('is_quadratic', True)])
def test_replaced_mesh_getter_keeps_complete_legacy_beam_result(monkeypatch, descriptor, preimport):
    from anymesher.mesh import Mesh
    original = getattr(Mesh, descriptor)
    calls = []
    def changed(mesh):
        calls.append(descriptor)
        if mesh.beams:
            mesh.quads.clear()
        return original.fget(mesh)
    monkeypatch.setattr(Mesh, descriptor, property(changed))
    implementation = mapped
    if preimport:
        spec = importlib.util.spec_from_file_location('anymesher._mapped_custom_getter_test', mapped.__file__)
        implementation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(implementation)
    model, edges = plate()
    options = dict(target_size=.5, order='quadratic', beam_edges=edges[:2],
                   beam_offsets=dict.fromkeys(edges[:2], .125))
    actual = implementation.generate_mesh(model, **options)
    actual_calls = list(calls)
    calls.clear()
    monkeypatch.setattr(implementation, '_build_beam', legacy_build_beam)
    expected = implementation.generate_mesh(model, **options)
    assert calls == actual_calls
    assert_exact_mesh(actual, expected)
    assert not implementation._native_beam_phase(model, actual)


def test_added_mesh_getattribute_callback_declines_index(monkeypatch):
    from anymesher.mesh import Mesh
    original = Mesh.__getattribute__
    def custom(mesh, name):
        return original(mesh, name)
    monkeypatch.setattr(Mesh, '__getattribute__', custom)
    assert not mapped._native_beam_phase(GeometryModel(), Mesh())
