"""One-extraction batching retains actual owner/registry/literal frontiers."""
from dataclasses import replace
from types import MethodType
import numpy as np
import pytest
import anygeometry as owner
from test_physical_mixed_repair_batch import chart_fixture
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher.mesh import Mesh
from anymesher.seeding import Seeding
from anymesher.refinement import SizeField
from anymesher.meshing_view import GeometryMeshingView
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.hybrid import _ensure_edge_registry
from anymesher.errors import MeshError


@pytest.fixture
def boundary(chart_fixture):
    model,chart=chart_fixture
    binding=chart.region_binding
    chords=binding._curvilinear_state['chords']
    targets={}
    for interval in chords.intervals:
        for parameter,point in zip(interval.parameter_range,interval.xyz):
            targets.setdefault(interval.edge.id,{})[float(parameter)]=tuple(point)
    seeding=Seeding(divisions={edge:4 for edge in targets})
    mesh=Mesh(geometry_model_id=model.model_id,geometry_revision=model.revision,seeding=seeding)
    view=GeometryMeshingView(model); registry=GlobalEdgeBoundaryRegistry(view)
    _ensure_edge_registry(model,view,mesh,registry,tuple(targets),seeding,SizeField(model,.25),
                          'linear',owner_stations=targets)
    return model,binding,mesh,registry


def test_native_pair_batch_equals_callback_original_with_two_complete_receipt_checks(boundary,monkeypatch):
    model,binding,mesh,registry=boundary
    original=owner.validate_prepared_material_boundary_chords_binding
    calls=[]
    def counted(*args,**kwargs):
        calls.append(args[2]); return original(*args,**kwargs)
    monkeypatch.setattr(owner,'validate_prepared_material_boundary_chords_binding',counted)
    before=owner.to_dict(model),repr(mesh.nodes_of_edge),repr(mesh.nodes)
    batched=binding.boundary_intervals(mesh,registry)
    assert len(calls)==2
    calls.clear(); phases=[]
    def check(phase): phases.append(phase); return False
    serial=binding.boundary_intervals(mesh,registry,check)
    paths=sum(map(len,binding.region.boundaries))
    assert len(calls)==2*paths and phases
    assert batched==serial
    assert before==(owner.to_dict(model),repr(mesh.nodes_of_edge),repr(mesh.nodes))
    calls.clear()
    path=binding.region.boundaries[0][0]
    binding.path_chain(path,mesh,registry)
    assert len(calls)==1  # public/direct source-chain calls still authenticate
    binding._boundary_source_chain(path,mesh,registry)
    assert len(calls)==2


@pytest.mark.parametrize('change',('replace_receipt','xyz','registry_owner','extra_station'))
def test_final_owner_frontier_mutations_refuse(boundary,monkeypatch,change):
    model,binding,mesh,registry=boundary
    original=MaterialRegionBinding.validate; calls=0
    saved=binding._curvilinear_state['chords']
    nodes=dict(mesh.nodes)
    def validate(self,check=None):
        nonlocal calls
        original(self,check); calls+=1
        if calls==2:
            if change=='replace_receipt': self._curvilinear_state['chords']=replace(saved)
            elif change=='xyz': mesh.nodes[next(iter(mesh.nodes))]=(99.,99.,99.)
            elif change=='registry_owner': registry.view=GeometryMeshingView(owner.GeometryModel())
            else: registry.register(next(iter(mesh.nodes_of_edge)),.125,node_id=max(mesh.nodes)+1)
    monkeypatch.setattr(MaterialRegionBinding,'validate',validate)
    try:
        with pytest.raises((MeshError,owner.GeometryError)):
            binding.boundary_intervals(mesh,registry)
        assert calls==2
    finally:
        binding._curvilinear_state['chords']=saved
        mesh.nodes.clear(); mesh.nodes.update(nodes)


@pytest.mark.parametrize('hook',('entries','view','nodes'))
def test_custom_hooks_keep_original_chain_validation(boundary,monkeypatch,hook):
    _model,binding,mesh,registry=boundary
    if hook=='entries':
        original=registry.entries
        registry.entries=lambda edge=None,method=original:method(edge)
    elif hook=='view':
        original=registry.view.assert_current
        registry.view.assert_current=lambda geometry,method=original:method(geometry)
    else:
        class Nodes(dict): pass
        mesh.nodes=Nodes(mesh.nodes)
    original=owner.validate_prepared_material_boundary_chords_binding; calls=[]
    def counted(*args,**kwargs): calls.append(1); return original(*args,**kwargs)
    monkeypatch.setattr(owner,'validate_prepared_material_boundary_chords_binding',counted)
    binding.boundary_intervals(mesh,registry)
    assert len(calls)==2*sum(map(len,binding.region.boundaries))


def test_callback_exception_identity_is_unchanged(boundary):
    _model,binding,mesh,registry=boundary
    failure=RuntimeError('boundary caller cancellation')
    def stop(phase): raise failure
    with pytest.raises(RuntimeError) as caught:
        binding.boundary_intervals(mesh,registry,stop)
    assert caught.value is failure


def test_genuine_multi_region_receipt_ignores_other_region_source_edges(chart_fixture):
    model,chart=chart_fixture
    before=owner.to_dict(model)
    collection=owner.query_material_surface_regions(model,tuple(model.faces))
    region=next(region for region in collection.regions
                if any(face.id==chart.face_id for face in region.faces))
    binding=MaterialRegionBinding(model,collection,region,chart.face_id,
                                  geometry_approximation_error=.1)
    edges=sorted({int(path.source_edge) for item in collection.regions
                  for loop in item.boundaries for path in loop})
    intervals=tuple((model.handle('edge',edge),i/8,(i+1)/8) for edge in edges for i in range(8))
    chords=owner.query_prepared_material_boundary_chords(model,collection,intervals,chord_error=.1)
    owner.validate_prepared_material_boundary_chords_binding(model,collection,chords)
    selected={int(path.source_edge) for loop in region.boundaries for path in loop}
    assert any(interval.edge.id not in selected for interval in chords.intervals)
    assert any(not any(item.face.id==chart.face_id for item in interval.occurrences)
               for interval in chords.intervals if interval.edge.id not in selected)
    binding._curvilinear_state['chords']=chords
    targets={}
    for interval in chords.intervals:
        for parameter,point in zip(interval.parameter_range,interval.xyz):
            targets.setdefault(interval.edge.id,{})[float(parameter)]=tuple(point)
    seeding=Seeding(divisions={edge:8 for edge in edges})
    mesh=Mesh(geometry_model_id=model.model_id,geometry_revision=model.revision,seeding=seeding)
    view=GeometryMeshingView(model); registry=GlobalEdgeBoundaryRegistry(view)
    _ensure_edge_registry(model,view,mesh,registry,edges,seeding,SizeField(model,.25),
                          'linear',owner_stations=targets)
    batched=binding.boundary_intervals(mesh,registry)
    serial=binding.boundary_intervals(mesh,registry,lambda phase:False)
    assert batched==serial and batched
    assert {receipt.source_edge for receipt in batched}==selected
    assert before==owner.to_dict(model)


@pytest.mark.parametrize('nested',('node','parameter','point','owner'))
def test_nested_registry_hooks_keep_original_frontiers(boundary,monkeypatch,nested):
    _model,binding,mesh,registry=boundary
    entry=registry.entries()[0]; hooks=[]
    if nested=='node':
        class HookInt(int):
            def __int__(self): hooks.append('int'); return int.__int__(self)
        object.__setattr__(entry,'node_id',HookInt(entry.node_id))
    elif nested=='parameter':
        class HookFloat(float):
            def __float__(self): hooks.append('float'); return float.__float__(self)
        object.__setattr__(entry.key,'parameter',HookFloat(entry.key.parameter))
    elif nested=='point':
        class HookPoint(np.ndarray):
            def __iter__(self): hooks.append('point'); return super().__iter__()
        object.__setattr__(entry,'point',entry.point.view(HookPoint))
    else:
        class HookOwner:
            def __repr__(self):
                raise AssertionError('batched registry pin executed owner repr hook')
        object.__setattr__(entry,'owners',(HookOwner(),))
    original=owner.validate_prepared_material_boundary_chords_binding; calls=[]
    def counted(*args,**kwargs): calls.append(1); return original(*args,**kwargs)
    monkeypatch.setattr(owner,'validate_prepared_material_boundary_chords_binding',counted)
    receipts=binding.boundary_intervals(mesh,registry)
    assert receipts and len(calls)==2*sum(map(len,binding.region.boundaries))
    if nested in ('node','parameter'): assert hooks
    else: assert not hooks


@pytest.mark.parametrize('container',('outer','loop'))
def test_boundary_container_subclasses_keep_original_frontiers(boundary,monkeypatch,container):
    _model,binding,mesh,registry=boundary
    domain=binding.region.domain; original_boundaries=domain.boundaries; hooks=[]
    class HookTuple(tuple):
        def __getitem__(self,key):
            hooks.append(key)
            return super().__getitem__(key)
    changed=(HookTuple(original_boundaries) if container=='outer' else
             (HookTuple(original_boundaries[0]),*original_boundaries[1:]))
    original=owner.validate_prepared_material_boundary_chords_binding; calls=[]
    def counted(*args,**kwargs): calls.append(1); return original(*args,**kwargs)
    monkeypatch.setattr(owner,'validate_prepared_material_boundary_chords_binding',counted)
    try:
        object.__setattr__(domain,'boundaries',changed)
        receipts=binding.boundary_intervals(mesh,registry)
        assert receipts and hooks
        assert len(calls)==2*sum(map(len,original_boundaries))
    finally:
        object.__setattr__(domain,'boundaries',original_boundaries)
