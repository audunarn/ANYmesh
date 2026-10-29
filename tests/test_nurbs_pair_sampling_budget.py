"""New-pair sampling budgets are proposals; original owner criteria persist."""
from fractions import Fraction as Q
import numpy as np
import pytest
import anygeometry as owner
from test_physical_mixed_repair_batch import chart_fixture
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.mesh import Mesh
from anymesher.seeding import Seeding
from anymesher.refinement import SizeField
from anymesher.meshing_view import GeometryMeshingView
from anymesher.hybrid import _ensure_edge_registry


def _event(phase,**details):
    # Optional, transparent root-harness evidence. No callback enters geometry.
    import json,os
    from pathlib import Path
    from time import perf_counter
    target=os.environ.get('ANYMESH_PAIR_SAMPLING_LOG')
    if target:
        with Path(target).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(phase=phase,clock=perf_counter(),**details))+'\n')


def production_operands(model):
    from anymesher._material_region_binding import prepare_material_regions
    from anymesher.native_v2 import NativeMeshingOptions
    from anymesher.seeding import solve_seeding
    from anymesher.hybrid import _active_edges,_station_parameters
    faces=tuple(model.faces)
    size_field=SizeField(model,.25,active_faces=faces)
    edges=_active_edges(model,faces,())
    _event('production.seeding.begin',edges=len(edges))
    seeding=solve_seeding(model,size_field=size_field,overrides={},edge_ids=edges,
        unstructured_face_ids=faces,maximum_adjacent_growth=1.5)
    _event('production.seeding.end',divisions={int(edge):int(seeding[edge]) for edge in edges})
    _event('production.bindings.begin')
    grouped=prepare_material_regions(model,faces,{face:(face,) for face in faces},
        order='linear',native_options=NativeMeshingOptions(point_placement='frontal_delaunay',
        metric_mode='isotropic_spatial'),supplied_seeding=False,edge_overrides={},
        geometry_approximation_error=.0025)
    bindings=tuple({id(binding):binding for binding in grouped.values()}.values())
    _event('production.bindings.end',regions=len(bindings),
        charts=[len(binding.collection.source.charts) for binding in bindings])
    initial={edge:tuple(_station_parameters(model,edge,int(seeding[edge]),size_field)) for edge in edges}
    return bindings,size_field,seeding,initial


def test_pair_common_sampling_union_and_original_error_receipts(chart_fixture,monkeypatch):
    model,_chart=chart_fixture
    before=owner.to_dict(model)
    sampled=[]; original=owner.query_prepared_material_boundary_sampling
    original_chords=owner.query_prepared_material_boundary_chords
    def traced(name,query,args,kwargs):
        _event(name+'.begin',charts=len(args[1].source.charts),intervals=len(args[2]),
            criterion=kwargs['chord_error'])
        try:
            result=query(*args,**kwargs)
        except BaseException as error:
            _event(name+'.error',error=type(error).__name__)
            raise
        intervals=result.chords.intervals if name=='sampling' else result.intervals
        _event(name+'.end',intervals=len(intervals))
        return result
    def measured(*args,**kwargs):
        sampled.append(kwargs['chord_error'])
        return traced('sampling',original,args,kwargs)
    def measured_chords(*args,**kwargs):
        return traced('chords',original_chords,args,kwargs)
    monkeypatch.setattr(owner,'query_prepared_material_boundary_sampling',measured)
    monkeypatch.setattr(owner,'query_prepared_material_boundary_chords',measured_chords)
    bindings,size_field,seeding,initial=production_operands(model)
    assert all(binding.collection.source.charts==binding.region.sources
               and len(binding.collection.regions)==1 for binding in bindings)
    from anymesher._material_region_binding import _prepare_curvilinear_station_union
    original_seed=MaterialRegionBinding.prepare_curvilinear_seed
    requested_union={}; requests={}; first_pass=[]
    def observed_seed(binding,parameters,cancellation_check=None,*,refine=True):
        phase='refine' if refine else 'union'
        if refine:
            first_pass.append(binding.representative)
            for edge,values in parameters.items():
                assert set(initial[edge]) <= set(values)
                assert set(requested_union.get(edge,())) <= set(values)
        _event('region.'+phase+'.begin',face=binding.representative,
            stations=sum(map(len,parameters.values())))
        targets=original_seed(binding,parameters,cancellation_check,refine=refine)
        _event('region.'+phase+'.end',face=binding.representative,
            stations=sum(map(len,targets.values())))
        assert binding.geometry_approximation_error == .0025
        assert binding._curvilinear_state['chords'].chord_error == .0025
        if refine:
            assert binding._curvilinear_state['sampling'].chord_error == .0025/4
            for edge,values in targets.items():
                assert set(parameters[edge]) <= set(values)
                requested_union.setdefault(edge,set()).update(values)
            requests[id(binding)]=targets
        else:
            for edge,values in targets.items():
                assert set(requests[id(binding)][edge]) <= set(values)==requested_union[edge]
            owner.validate_prepared_material_boundary_chords_binding(
                model,binding.collection,binding._curvilinear_state['chords'])
        return targets
    monkeypatch.setattr(MaterialRegionBinding,'prepare_curvilinear_seed',observed_seed)
    stations=_prepare_curvilinear_station_union(model,bindings,seeding,size_field)
    union=requested_union
    assert len(bindings)==4 and sampled==[.0025/4]*4
    assert len(first_pass)==4
    first=next(binding for binding in bindings if binding.representative==first_pass[0])
    assert sum(map(len,first.region.boundaries))==min(
        sum(map(len,binding.region.boundaries)) for binding in bindings)
    mesh=Mesh(geometry_model_id=model.model_id,geometry_revision=model.revision,seeding=seeding)
    view=GeometryMeshingView(model); registry=GlobalEdgeBoundaryRegistry(view)
    _ensure_edge_registry(model,view,mesh,registry,tuple(sorted(stations)),seeding,size_field,
        'linear',owner_stations=stations)
    for binding in bindings:
        binding.prepare_curvilinear_chords(mesh,registry)
        assert binding._curvilinear_state['chords'].chord_error == .0025
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
    shared={int(path.source_edge) for binding in bindings for loop in binding.region.boundaries
            for path in loop if type(path.curve) is NurbsSurfaceIntersectionCurve}
    assert len(shared)>=2
    for edge in shared:
        assert len(model.faces_using_edge(edge))==4
        entries=registry.entries(edge)
        assert tuple(entry.node_id for entry in entries)==tuple(mesh.nodes_of_edge[edge])
        assert len(set(entry.node_id for entry in entries))==len(entries)
        assert set(entry.key.parameter for entry in entries)==union[edge]
        for entry in entries:
            assert tuple(mesh.nodes[entry.node_id])==stations[edge][entry.key.parameter]
    assert before==owner.to_dict(model)


def test_ordinary_nurbs_sampling_preserves_original_error(monkeypatch):
    from test_nurbs_curvilinear_consumer import patch_model
    model,face,edges=patch_model()
    collection=owner.query_material_surface_regions(model,(face,))
    binding=MaterialRegionBinding(model,collection,collection.regions[0],face,
        geometry_approximation_error=.01)
    observed=[]; original=owner.query_prepared_material_boundary_sampling
    def measured(*args,**kwargs):
        observed.append(kwargs['chord_error'])
        return original(*args,**kwargs)
    monkeypatch.setattr(owner,'query_prepared_material_boundary_sampling',measured)
    binding.prepare_curvilinear_seed({edge:(0.,.5,1.) for edge in edges})
    assert observed==[.01] and binding._curvilinear_state['chords'].chord_error==.01


def test_genuine_owner_refined_original_boundary_cell_proof(chart_fixture):
    # Exact saved failing cell UV (source-cell-failure.json, 024629-fa8ed2).
    # Boundary-edge bisection is a component diagnostic, not source-complex coverage
    # or the final public mesh. Every child uses the unchanged .0025 criterion.
    from anygeometry.regular_source_reference import SourceNode,SourceSide
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    model,chart=chart_fixture
    support=model.faces[chart.face_id].surface
    uv=tuple(tuple(map(Q,row)) for row in ((0.,0.),(.05780572894257695,0.),
                                           (.0162014012693543,.014297904569265012)))
    midpoint=lambda a,b:tuple((x+y)/2 for x,y in zip(a,b))
    a,b,c=uv; ab=midpoint(a,b)
    children=((a,ab,c),(ab,b,c))
    for child_index,child in enumerate(children):
        xyz=tuple(tuple(float(x) for x in support.exact_point(*point)) for point in child)
        centre=tuple(sum(point[i] for point in child)/3 for i in (0,1))
        centre_xyz=tuple(sum(Q(point[i]) for point in xyz)/3 for i in range(3))
        nodes=tuple(SourceNode(('diagnostic-child',child_index,i),tuple((x,x) for x in point))
                    for i,point in enumerate(child))
        for side in range(3):
            proof,_pieces=certify_fan_subtriangle(support,SourceSide(nodes[side],nodes[(side+1)%3]),
                centre,(xyz[side],xyz[(side+1)%3]),centre_xyz,orientation=1,surface_error=Q(.0025))
            assert proof<=Q(.0025)


def test_pair_sampling_failure_propagates_without_publishing_candidate(chart_fixture,monkeypatch):
    model,chart=chart_fixture
    collection=owner.query_material_surface_regions(model,(chart.face_id,))
    binding=MaterialRegionBinding(model,collection,collection.regions[0],chart.face_id,
        geometry_approximation_error=.0025)
    edges={int(path.source_edge) for loop in binding.region.boundaries for path in loop}
    before=owner.to_dict(model),dict(binding._curvilinear_state)
    failure=RuntimeError('sampling operation cancelled')
    def stop(*args):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        binding.prepare_curvilinear_seed({edge:(0.,.5,1.) for edge in edges},stop)
    assert caught.value is failure
    assert before==(owner.to_dict(model),binding._curvilinear_state)


def test_proposal_helpers_preserve_legacy_positions_and_input_station_sets():
    from types import SimpleNamespace
    from anymesher._material_region_binding import _pair_sampling_order,_pair_sampling_parameters
    # Isolated ordering/set-union policy only: no fake result reaches an owner
    # query or establishes the genuine-route tests' geometry authority.
    def candidate(ident,count,pair):
        return SimpleNamespace(representative=ident,region=SimpleNamespace(boundaries=(tuple(range(count)),)),
            _uses_pair_boundary_sampling=lambda:pair)
    outer=candidate(3,6,True); legacy=candidate(4,1,False); inner=candidate(5,2,True)
    assert _pair_sampling_order((outer,legacy,inner))==(inner,legacy,outer)
    own={7:(0.,.5,1.),8:(0.,1.)}; previous={7:{.25,.75},99:{.125}}
    before=repr(own),repr(previous)
    assert _pair_sampling_parameters(legacy,own,previous) is own
    assert _pair_sampling_parameters(inner,own,previous)=={7:(0.,.25,.5,.75,1.),8:(0.,1.)}
    assert before==(repr(own),repr(previous))
