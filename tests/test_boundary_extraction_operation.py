"""Real current PlaneNurbs/NN chord extraction, never sampled authority."""
from anymesher.hybrid import _LoopBoundary
import numpy as np
import pytest
import anygeometry as owner
from anymesher.mesh import Mesh
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.meshing_view import GeometryMeshingView
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher._material_input_provenance import _TERMS
from anymesher.errors import MeshError
from test_plane_curvilinear_consumer import plane_disk
from test_physical_mixed_repair_batch import chart_fixture


def registered(binding):
    model=binding.geometry;chords=binding._curvilinear_state['chords']
    mesh=Mesh(geometry_model_id=model.model_id,geometry_revision=model.revision)
    registry=GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    slots={};vertices={};next_node=1
    for interval in chords.intervals:
        if not any(row.face.id==binding.representative for row in interval.occurrences):continue
        edge=model.edges[interval.edge.id]
        for parameter,point in zip(interval.parameter_range,interval.xyz):
            key=(edge.id,float(parameter))
            if key in slots:continue
            vertex=edge.start if parameter==0 else edge.end if parameter==1 else None
            node=vertices.get(vertex) if vertex is not None else None
            if node is None:
                node=next_node;next_node+=1
                if vertex is not None:vertices[vertex]=node
            slots[key]=node;mesh.nodes[node]=tuple(map(float,point))
            registry.register(edge.id,float(parameter),node_id=node,point=np.asarray(point,float))
    for edge in sorted({key[0] for key in slots}):
        mesh.nodes_of_edge[edge]=[slots[key] for key in sorted(slots) if key[0]==edge]
    mesh.node_of_vertex.update(vertices)
    return mesh,registry


@pytest.fixture(scope='module')
def plane_binding():
    model,face,edges=plane_disk();regions=owner.query_material_surface_regions(model,(face,))
    chords=owner.query_prepared_material_boundary_chords(model,regions,
        tuple((model.handle('edge',edge),0.,1.) for edge in edges),chord_error=1.)
    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,geometry_approximation_error=1.)
    binding._curvilinear_state['chords']=chords
    return binding


def assert_equivalent(binding):
    mesh,registry=registered(binding);before=owner.to_dict(binding.geometry)
    extract,finish=binding._prepare_boundary_source_chains(mesh,registry)
    assert finish is not None
    for loop in binding.region.boundaries:
        for path in loop:
            a=extract(path,mesh,registry);b=binding._boundary_source_chain(path,mesh,registry)
            assert a[0]==b[0]
            np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
    finish();assert owner.to_dict(binding.geometry)==before


def test_plane_current_literal_extraction_equals_fresh(plane_binding):assert_equivalent(plane_binding)

def test_nn_current_literal_extraction_equals_fresh(chart_fixture):assert_equivalent(chart_fixture[1].region_binding)


@pytest.mark.parametrize('change',('receipt','registry','mesh','source','runtime'))
def test_final_guard_refuses_changed_context(plane_binding,monkeypatch,change):
    binding=plane_binding;mesh,registry=registered(binding)
    extract,finish=binding._prepare_boundary_source_chains(mesh,registry);assert finish
    chord=binding._curvilinear_state['chords'];restore=lambda:None
    if change=='receipt':
        old=chord.chord_error;object.__setattr__(chord,'chord_error',old*2)
        restore=lambda:object.__setattr__(chord,'chord_error',old)
    elif change=='registry':
        from dataclasses import replace
        key,entry=next(iter(registry._entries.items()))
        point=entry.point.copy();point[0]+=1
        changed=replace(entry,point=point)
        registry._entries[key]=changed
        registry._by_edge[key.edge_id][key.parameter]=changed
    elif change=='mesh':mesh.nodes[min(mesh.nodes)]=(99.,99.,99.)
    elif change=='source':
        face=binding.geometry.faces[binding.representative];old=face.metadata
        object.__setattr__(face,'metadata',{'changed':True});restore=lambda:object.__setattr__(face,'metadata',old)
    else:monkeypatch.setattr(MaterialRegionBinding,'path_chain',lambda *a:None)
    try:
        with pytest.raises((owner.GeometryError,MeshError)):finish()
    finally:restore()


def test_callback_route_stays_fresh_and_exception_identity(plane_binding):
    binding=plane_binding;mesh,registry=registered(binding)
    extract,finish=binding._prepare_boundary_source_chains(mesh,registry,lambda phase:None)
    assert finish is None and extract==binding._boundary_source_chain
    token=RuntimeError('actual callback')
    def stop(phase):raise token
    with pytest.raises(RuntimeError) as caught:binding.boundary_intervals(mesh,registry,stop)
    assert caught.value is token


def test_payload_equals_forced_fresh_route(plane_binding,monkeypatch):
    binding=plane_binding;model=binding.geometry;mesh,registry=registered(binding)
    loops=tuple(_LoopBoundary(node_ids=tuple(n),midside_ids=(),uv=uv,segment_specs=tuple(sp)) for n,uv,sp in binding.loops(mesh,registry,()))
    edges={p.source_edge for loop in binding.region.boundaries for p in loop}
    context={'model_id':str(model.model_id),'revision':model.revision,'edges':{
        edge:{**{term:True for term in _TERMS},'default_seeding':False,'selected':False} for edge in edges}}
    emitted={'transform':np.eye(2),'chart_loops':tuple(l.uv.copy() for l in loops),
        'chart_constraints':(),'chart_pinned':np.empty((0,2))}
    actual=binding.input_provenance(mesh,registry,loops,(),context,emitted).to_dict()
    monkeypatch.setattr(MaterialRegionBinding,'_prepare_boundary_source_chains',
        lambda self,mesh,registry,check=None:(self._boundary_source_chain,None))
    expected=binding.input_provenance(mesh,registry,loops,(),context,emitted).to_dict()
    assert actual==expected


def test_unknown_effective_method_stays_on_fresh_path(plane_binding,monkeypatch):
    binding=plane_binding;mesh,registry=registered(binding)
    token=RuntimeError('custom interior callback')
    def custom(self,*args,**kwargs):raise token
    monkeypatch.setattr(MaterialRegionBinding,'interior',custom)
    extract,finish=binding._prepare_boundary_source_chains(mesh,registry)
    assert finish is None and extract==binding._boundary_source_chain
    # The actual custom behavior remains reachable; it is not bypassed by a
    # previously canonical class identity or a missing cancellation callback.
    with pytest.raises(RuntimeError) as caught:binding.interior(mesh,registry)
    assert caught.value is token


@pytest.mark.parametrize('timing',('entry','final'))
def test_hostile_default_repr_never_runs_or_authorizes_stale_receipt(plane_binding,timing):
    binding=plane_binding;mesh,registry=registered(binding)
    chord=binding._curvilinear_state['chords'];old_error=chord.chord_error;calls=[]
    class Hostile:
        def __repr__(self):
            calls.append('repr');object.__setattr__(chord,'chord_error',old_error*2)
            return 'None'
        def __eq__(self,other):
            calls.append('eq');raise AssertionError('user equality must not run')
    function=MaterialRegionBinding._prepare_boundary_source_chains
    old_defaults=function.__defaults__
    if timing=='final':
        extract,finish=binding._prepare_boundary_source_chains(mesh,registry)
        assert finish is not None
    try:
        function.__defaults__=(Hostile(),)
        if timing=='entry':
            extract,finish=binding._prepare_boundary_source_chains(mesh,registry,None)
            assert finish is None and extract==binding._boundary_source_chain
        else:
            with pytest.raises(MeshError,match='runtime'):finish()
        assert calls==[] and chord.chord_error==old_error
    finally:
        function.__defaults__=old_defaults
        object.__setattr__(chord,'chord_error',old_error)


def test_mutable_literal_defaults_are_snapshotted_without_user_equality():
    from anymesher._material_region_binding import _boundary_callable_pin
    data={'values':[1,{'flag':False}]}
    def function(value=data):pass
    before=_boundary_callable_pin(function)
    data['values'][1]['flag']=True
    assert _boundary_callable_pin(function)!=before
    function.__kwdefaults__={'setting':{'items':[1]}}
    before=_boundary_callable_pin(function)
    function.__kwdefaults__['setting']['items'].append(2)
    assert _boundary_callable_pin(function)!=before
