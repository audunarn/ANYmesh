"""Boundary publication tests using actual live owner-issued transfer receipts."""
from copy import deepcopy
import pytest
from anymesher.errors import MeshError
from anymesher.material_source_transfer import validate_material_source_transfer_for_mesh
from anymesher._source_boundary_publication import source_boundaries
from test_material_source_transfer_consumer import prepared_mesh


def test_actual_receipt_boundary_transport_has_only_source_edge_ids():
    source, prepared, mesh, _face = prepared_mesh()
    receipt = validate_material_source_transfer_for_mesh(source, prepared, mesh)
    before = deepcopy(mesh.nodes), deepcopy(mesh.nodes_of_edge)
    tables, registry, traces, vertices, _maps = source_boundaries(source, prepared, mesh,
        receipt, {vertex:vertex for vertex in source.vertices})
    assert not traces and not vertices
    assert set(tables['nodes_of_edge']) <= set(source.edges)
    assert {row.key.edge_id for row in registry.entries()} <= set(source.edges)
    assert tables['nodes_of_edge'] == mesh.nodes_of_edge
    assert mesh.nodes_of_edge == before[1]
    assert all(tuple(mesh.nodes[node]) == tuple(point) for node,point in before[0].items())


def test_actual_receipt_cannot_transport_changed_station_xyz():
    source, prepared, mesh, _face = prepared_mesh()
    receipt = validate_material_source_transfer_for_mesh(source, prepared, mesh)
    node = next(iter(mesh.nodes))
    mesh.nodes[node] = tuple(float(v)+.1 for v in mesh.nodes[node])
    with pytest.raises(MeshError, match='literal owner correspondence'):
        source_boundaries(source, prepared, mesh, receipt,
            {vertex:vertex for vertex in source.vertices})


def closure_publication_operands(*, establish_publication=True):
    import anygeometry as owner
    from anymesher.hybrid import generate_hybrid_mesh_result
    from test_nurbs_curvilinear_consumer import patch_model, options
    source, face, _edges = patch_model(warp=0.)
    sheet = next(iter(source.sheets))
    closure = owner.extract_model_closure(source, (source.handle('sheet',sheet),))
    request = options()
    request.update(target_size=2., recombine=False, geometry_approximation_error=.0025)
    prepared = owner.clone_prepared_geometry(closure.working_model)
    mesh = generate_hybrid_mesh_result(prepared,
        mutation_policy='working_copy', **request).mesh
    if establish_publication:
        from anymesher.material_source_transfer import _publish_identity_source_transfer
        _publish_identity_source_transfer(closure.working_model, prepared, mesh,
                                          mesh.boundary_registry)
    return source, closure, mesh


@pytest.mark.parametrize('attack', ('thickness','nested_nodes','registry_point','registry_dynamic','return_mutation','readonly_base_view',
                                  'immutable_shape','immutable_dtype','zero_stride_view',
                                  'immutable_dtype_zero_stride','reversed_dtype_zero_stride',
                                  'noncontiguous_dtype_zero_stride'))
def test_actual_publisher_first_callback_restores_nested_aliases_and_prior_authority(attack):
    import numpy as np
    from anymesher.material_source_transfer import (publish_model_closure_mesh_to_source,
        _IDENTITY_PUBLICATIONS, _publication_pin, _validate_identity_source_publication,
        _publish_identity_source_transfer)
    source, closure, mesh = closure_publication_operands(establish_publication=False)
    node = next(iter(mesh.nodes))
    mesh.publication_probe = {'nodes': [[mesh.nodes[node]]]}
    readonly_base = np.array([1.,2.,3.])
    readonly_view = readonly_base.view()
    readonly_base.setflags(write=False)
    readonly_view.setflags(write=False)
    # Base-before-view insertion deliberately defeats reversed DFS ordering.
    mesh.publication_probe['shared_arrays'] = [readonly_base, readonly_view]
    immutable_bytes = np.array([1.,2.,3.]).tobytes()
    immutable = np.frombuffer(immutable_bytes, dtype=float)
    stride_base = np.array([4.,5.,6.])
    stride_view = stride_base.view()
    mesh.publication_probe['metadata_arrays'] = [immutable, stride_base, stride_view]
    reversed_view = immutable[::-1]
    noncontiguous_base = np.array([4.,5.,6.,7.,8.])
    noncontiguous_view = noncontiguous_base[::2]
    mesh.publication_probe['noncontiguous_arrays'] = [reversed_view, noncontiguous_base, noncontiguous_view]
    reversed_metadata = reversed_view.dtype, reversed_view.shape, reversed_view.strides, reversed_view.flags.writeable
    noncontiguous_metadata = noncontiguous_view.dtype, noncontiguous_view.shape, noncontiguous_view.strides, noncontiguous_view.flags.writeable
    immutable_metadata = immutable.dtype, immutable.shape, immutable.strides, immutable.flags.writeable
    stride_metadata = stride_view.dtype, stride_view.shape, stride_view.strides, stride_view.flags.writeable
    prepared = mesh.boundary_registry.view.source
    _publish_identity_source_transfer(closure.working_model, prepared, mesh,
                                      mesh.boundary_registry)
    registry = mesh.boundary_registry
    entry = registry.entries()[0]
    point = entry.point
    point_bytes, point_writeable = point.tobytes(), point.flags.writeable
    nodes, thickness = mesh.nodes, mesh.thickness_of_face
    node = next(iter(nodes))
    held_node = nodes[node]
    held_values = tuple(held_node)
    held_face_cells = next(iter(mesh.elements_of_face.values()))
    old_cells = list(held_face_cells)
    held_probe, held_nested = mesh.publication_probe, mesh.publication_probe['nodes'][0]
    before_pin = _publication_pin(mesh)
    previous = _IDENTITY_PUBLICATIONS.get(id(mesh))
    class Attack(BaseException):
        pass
    error = Attack('callback failure')
    calls = []
    def mutate(phase):
        calls.append(phase)
        if attack == 'thickness':
            thickness[999999] = 19.
        elif attack in ('nested_nodes','return_mutation'):
            held_face_cells.append(999999)
            held_nested.append([17.])
            if isinstance(held_node, np.ndarray):
                held_node[0] += .1
            else:
                nodes[node] = (99.,99.,99.)
        elif attack == 'registry_point':
            point.setflags(write=True)
            point[0] += .1
            object.__setattr__(entry, 'point', np.array((8.,9.,10.)))
        elif attack == 'readonly_base_view':
            readonly_base.setflags(write=True)
            readonly_base[0] = 99.
            readonly_base.setflags(write=False)
        elif attack == 'immutable_shape':
            immutable.shape = (1,3)
        elif attack == 'immutable_dtype':
            immutable.dtype = np.uint8
        elif attack == 'zero_stride_view':
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                stride_view.strides = (0,)
        elif attack in ('immutable_dtype_zero_stride','reversed_dtype_zero_stride','noncontiguous_dtype_zero_stride'):
            import warnings
            target = (immutable if attack == 'immutable_dtype_zero_stride' else
                      reversed_view if attack == 'reversed_dtype_zero_stride' else noncontiguous_view)
            target.dtype = np.uint8 if target is immutable else np.int64
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                target.strides = (0,)
        else:
            registry.callback_inserted_state = {'bad': [entry]}
            registry._entries = {}
        if attack == 'return_mutation':
            return False
        raise error
    with pytest.raises((Attack,MeshError)) as caught:
        publish_model_closure_mesh_to_source(source, closure, mesh, cancellation_check=mutate)
    if attack != 'return_mutation':
        assert caught.value is error
    assert calls  # Attack reached the actual first owner qualification callback.
    assert mesh.nodes is nodes and mesh.thickness_of_face is thickness
    assert nodes[node] is held_node and tuple(held_node) == held_values
    assert held_face_cells == old_cells
    assert mesh.publication_probe is held_probe and held_probe['nodes'][0] is held_nested
    assert held_nested == [held_node]
    assert mesh.publication_probe['shared_arrays'][0] is readonly_base
    assert mesh.publication_probe['shared_arrays'][1] is readonly_view
    assert readonly_view.base is readonly_base
    assert tuple(readonly_base) == tuple(readonly_view) == (1.,2.,3.)
    assert not readonly_base.flags.writeable and not readonly_view.flags.writeable
    assert mesh.publication_probe['metadata_arrays'][0] is immutable
    assert immutable.base is immutable_bytes
    assert (immutable.dtype,immutable.shape,immutable.strides,immutable.flags.writeable) == immutable_metadata
    assert immutable.tobytes() == immutable_bytes
    assert mesh.publication_probe['metadata_arrays'][1] is stride_base
    assert mesh.publication_probe['metadata_arrays'][2] is stride_view and stride_view.base is stride_base
    assert (stride_view.dtype,stride_view.shape,stride_view.strides,stride_view.flags.writeable) == stride_metadata
    assert tuple(stride_base) == tuple(stride_view) == (4.,5.,6.)
    assert mesh.publication_probe['noncontiguous_arrays'][0] is reversed_view
    assert reversed_view.base is immutable
    assert (reversed_view.dtype,reversed_view.shape,reversed_view.strides,reversed_view.flags.writeable) == reversed_metadata
    assert tuple(reversed_view) == (3.,2.,1.)
    assert mesh.publication_probe['noncontiguous_arrays'][1] is noncontiguous_base
    assert mesh.publication_probe['noncontiguous_arrays'][2] is noncontiguous_view
    assert noncontiguous_view.base is noncontiguous_base
    assert (noncontiguous_view.dtype,noncontiguous_view.shape,noncontiguous_view.strides,noncontiguous_view.flags.writeable) == noncontiguous_metadata
    assert tuple(noncontiguous_view) == (4.,6.,8.)
    assert mesh.boundary_registry is registry and entry.point is point
    assert point.tobytes() == point_bytes and point.flags.writeable == point_writeable
    assert not hasattr(registry, 'callback_inserted_state')
    assert _publication_pin(mesh) == before_pin
    assert _IDENTITY_PUBLICATIONS.get(id(mesh)) is previous
    assert previous is not None
    _validate_identity_source_publication(closure.working_model, previous[2], mesh, registry)


@pytest.mark.parametrize('association', ('edge_missing','edge_shell','sheet_extra'))
def test_actual_publisher_refuses_malformed_associations(association):
    from anymesher.material_source_transfer import publish_model_closure_mesh_to_source, _publication_pin
    source, closure, mesh = closure_publication_operands()
    if association == 'sheet_extra':
        mesh.elements_of_sheet[next(iter(mesh.elements_of_sheet))].append(999999)
    else:
        edge = next(iter(mesh.nodes_of_edge))
        mesh.elements_of_edge[edge] = [999999 if association == 'edge_missing' else next(iter(mesh.shells))]
    before = _publication_pin(mesh)
    with pytest.raises(MeshError):
        publish_model_closure_mesh_to_source(source, closure, mesh)
    assert _publication_pin(mesh) == before

def qualified_closure_publication_operands():
    """Real initial qualification, genuine owner clone, real generated cells."""
    import anygeometry as owner
    from anymesher import qualify_model_closure_source_publication
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.material_source_transfer import _publish_identity_source_transfer
    from test_nurbs_curvilinear_consumer import patch_model, options
    source, _face, _edges = patch_model(warp=0.)
    closure = owner.extract_model_closure(source, tuple(
        source.handle('sheet', sheet) for sheet in source.sheets))
    token = qualify_model_closure_source_publication(source, closure)
    prepared = owner.clone_prepared_geometry(closure.working_model)
    request = options()
    request.update(target_size=2., recombine=False, geometry_approximation_error=.0025)
    mesh = generate_hybrid_mesh_result(prepared, mutation_policy='working_copy', **request).mesh
    _publish_identity_source_transfer(closure.working_model, prepared, mesh, mesh.boundary_registry)
    return source, closure, prepared, mesh, token


def test_prequalified_publication_uses_both_live_source_and_work_clone_proofs(monkeypatch):
    import anygeometry as owner
    import anymesher.material_source_transfer as transfer
    source, closure, prepared, mesh, token = qualified_closure_publication_operands()
    before = owner.to_dict(source), owner.to_dict(closure.working_model), owner.to_dict(prepared)
    actual = transfer.validate_material_source_transfer_for_mesh
    checked_sources = []
    def observe(current, *args, **kwargs):
        result = actual(current, *args, **kwargs)
        checked_sources.append(current)
        return result
    monkeypatch.setattr(transfer, 'validate_material_source_transfer_for_mesh', observe)
    physical = dict(mesh.nodes), dict(mesh.tris), dict(mesh.quads)
    transfer.publish_model_closure_mesh_to_source(source, closure, mesh, qualification=token)
    assert source in checked_sources and closure.working_model in checked_sources
    assert mesh.geometry_model_id == source.model_id
    assert mesh.structural_preparation['qualified_s3']['status'] == 'ADMITTED'
    assert mesh.structural_preparation['qualified_s3']['authority_model']['source_model_id'] == str(source.model_id)
    assert (dict(mesh.nodes), dict(mesh.tris), dict(mesh.quads)) == physical
    assert (owner.to_dict(source), owner.to_dict(closure.working_model), owner.to_dict(prepared)) == before


@pytest.mark.parametrize('attack', ('forged','copied','serialized','source_edit','raw_source_edit','maps','selection','unqualified_work_clone'))
def test_prequalified_actual_publisher_refuses_forged_stale_or_substituted_inputs(attack):
    from copy import copy
    from dataclasses import replace
    from types import MappingProxyType
    import anygeometry as owner
    from anymesher.errors import MeshError
    from anymesher.material_source_transfer import (
        publish_model_closure_mesh_to_source, _publication_pin,
        _IDENTITY_PUBLICATIONS, _validate_identity_source_publication)
    source, closure, prepared, mesh, token = qualified_closure_publication_operands()
    original_work = closure.working_model
    previous = _IDENTITY_PUBLICATIONS[id(mesh)]
    before = _publication_pin(mesh)
    if attack == 'forged':
        token = object()
    elif attack == 'copied':
        token = copy(token)
    elif attack == 'serialized':
        import pickle
        token = pickle.loads(pickle.dumps(token))
    elif attack == 'raw_source_edit':
        face = next(iter(source.faces))
        source._faces[face] = replace(source.faces[face], metadata={'raw_source_edit': True})
    elif attack == 'source_edit':
        source.update_face_metadata(next(iter(source.faces)), external_edit=True)
    elif attack == 'maps':
        closure = replace(closure, source_to_work=MappingProxyType(dict(closure.source_to_work)))
    elif attack == 'selection':
        closure = replace(closure, source_handles=tuple(list(closure.source_handles)))
    else:
        closure = replace(closure, working_model=original_work.clone(preserve_identity=True))
    # External source edits belong to the caller and must not be overwritten.
    source_after_attack = owner.to_dict(source)
    with pytest.raises((MeshError, owner.GeometryError)):
        publish_model_closure_mesh_to_source(source, closure, mesh, qualification=token)
    assert _publication_pin(mesh) == before
    assert _IDENTITY_PUBLICATIONS[id(mesh)] is previous
    _validate_identity_source_publication(original_work, prepared, mesh, mesh.boundary_registry)
    assert owner.to_dict(source) == source_after_attack


def test_prequalified_publication_callback_baseexception_restores_prior_live_authority():
    import anygeometry as owner
    from anymesher.material_source_transfer import (
        publish_model_closure_mesh_to_source, _publication_pin,
        _IDENTITY_PUBLICATIONS, _validate_identity_source_publication)
    source, closure, prepared, mesh, token = qualified_closure_publication_operands()
    previous = _IDENTITY_PUBLICATIONS[id(mesh)]
    before = _publication_pin(mesh)
    nodes, registry = mesh.nodes, mesh.boundary_registry
    class Cancelled(BaseException):
        pass
    failure = Cancelled()
    def callback(_phase):
        mesh.nodes[next(iter(mesh.nodes))] = (9.,9.,9.)
        raise failure
    with pytest.raises(Cancelled) as caught:
        publish_model_closure_mesh_to_source(source, closure, mesh,
            qualification=token, cancellation_check=callback)
    assert caught.value is failure
    assert mesh.nodes is nodes and mesh.boundary_registry is registry
    assert _publication_pin(mesh) == before and _IDENTITY_PUBLICATIONS[id(mesh)] is previous
    _validate_identity_source_publication(closure.working_model, prepared, mesh, registry)



def test_prequalified_publisher_late_raw_source_edit_is_preserved_but_mesh_rolls_back():
    from dataclasses import replace
    import anygeometry as owner
    from anymesher.material_source_transfer import (
        publish_model_closure_mesh_to_source, _publication_pin,
        _IDENTITY_PUBLICATIONS, _validate_identity_source_publication)
    source, closure, prepared, mesh, token = qualified_closure_publication_operands()
    previous = _IDENTITY_PUBLICATIONS[id(mesh)]
    before = _publication_pin(mesh)
    nodes, registry, associations = mesh.nodes, mesh.boundary_registry, mesh.elements_of_face
    revision = source.revision
    face = next(iter(source.faces))
    changed = []
    def callback(phase):
        if phase == 'closure source publication complete':
            source._faces[face] = replace(source.faces[face], metadata={'external_late_edit': True})
            changed.append(owner.to_dict(source))
        return False
    with pytest.raises((MeshError, owner.GeometryError)):
        publish_model_closure_mesh_to_source(source, closure, mesh,
            qualification=token, cancellation_check=callback)
    assert changed and source.revision == revision
    assert owner.to_dict(source) == changed[-1]
    assert mesh.nodes is nodes and mesh.boundary_registry is registry
    assert mesh.elements_of_face is associations
    assert _publication_pin(mesh) == before and _IDENTITY_PUBLICATIONS[id(mesh)] is previous
    _validate_identity_source_publication(closure.working_model, prepared, mesh, registry)

@pytest.mark.parametrize('location', ('source_key','work_value','selector'))
@pytest.mark.parametrize('field', ('id','kind','model_id'))
def test_prequalified_publisher_detects_in_place_handle_mutation(location, field):
    from uuid import uuid4
    from anymesher.material_source_transfer import (
        publish_model_closure_mesh_to_source, _publication_pin,
        _IDENTITY_PUBLICATIONS, _validate_identity_source_publication)
    source, closure, prepared, mesh, token = qualified_closure_publication_operands()
    key, value = next(iter(closure.source_to_work.items()))
    target = {'source_key':key, 'work_value':value, 'selector':closure.source_handles[0]}[location]
    previous = _IDENTITY_PUBLICATIONS[id(mesh)]
    before = _publication_pin(mesh)
    replacement = {'id':target.id+1000, 'kind':'edge' if target.kind != 'edge' else 'face',
                   'model_id':uuid4()}[field]
    # Keep the actual map/tuple and actual handle objects: only raw fields change.
    object.__setattr__(target, field, replacement)
    with pytest.raises(MeshError, match='maps or selection changed'):
        publish_model_closure_mesh_to_source(source, closure, mesh, qualification=token)
    assert getattr(target, field) == replacement
    assert _publication_pin(mesh) == before and _IDENTITY_PUBLICATIONS[id(mesh)] is previous
    _validate_identity_source_publication(closure.working_model, prepared, mesh, mesh.boundary_registry)


@pytest.mark.parametrize('malformed', (None, {}, object()))
def test_public_closure_qualification_and_publisher_reject_malformed_closure_typed(malformed):
    import anygeometry as owner
    from anymesher import (Mesh, qualify_model_closure_source_publication,
                          publish_model_closure_mesh_to_source)
    source = owner.GeometryModel()
    with pytest.raises((MeshError, owner.GeometryError)):
        qualify_model_closure_source_publication(source, malformed)
    with pytest.raises((MeshError, owner.GeometryError)):
        publish_model_closure_mesh_to_source(source, malformed, Mesh())


def test_prequalified_publisher_late_in_place_selector_mutation_rolls_back_mesh():
    from anymesher.material_source_transfer import (
        publish_model_closure_mesh_to_source, _publication_pin,
        _IDENTITY_PUBLICATIONS, _validate_identity_source_publication)
    source, closure, prepared, mesh, token = qualified_closure_publication_operands()
    previous, before = _IDENTITY_PUBLICATIONS[id(mesh)], _publication_pin(mesh)
    handle = closure.source_handles[0]
    initial = handle.id
    def callback(phase):
        if phase == 'closure source publication complete':
            object.__setattr__(handle, 'id', initial+1000)
        return False
    with pytest.raises(MeshError, match='maps or selection changed'):
        publish_model_closure_mesh_to_source(source, closure, mesh,
            qualification=token, cancellation_check=callback)
    assert handle.id == initial+1000
    assert _publication_pin(mesh) == before and _IDENTITY_PUBLICATIONS[id(mesh)] is previous
    _validate_identity_source_publication(closure.working_model, prepared, mesh, mesh.boundary_registry)
