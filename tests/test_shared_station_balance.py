from fractions import Fraction as Q
from types import SimpleNamespace as NS
import pytest
import anygeometry as owner
from anymesher._material_region_binding import _station_balance_splits,_prepare_curvilinear_station_union,prepare_material_regions
from anymesher.errors import MeshError


def test_split_uses_exact_source_vertex_adjacency_not_proximity():
    geometry=NS(edges={1:NS(start=1,end=2),2:NS(start=2,end=3),3:NS(start=8,end=9)})
    targets={1:{0.:(0.,0.,0.),1.:(2.,0.,0.)},2:{0.:(2.,0.,0.),1.:(2.5,0.,0.)}}
    assert _station_balance_splits(geometry,targets,1.5)==((1,.5),)
    detached={1:targets[1],3:targets[2]}
    assert _station_balance_splits(geometry,detached,1.5)==()
    assert tuple(targets[1])==(0.,1.)


def test_same_edge_longer_interval_only_and_reversed_world_direction():
    geometry=NS(edges={1:NS(start=1,end=2)})
    targets={1:{0.:(3.,0.,0.),.5:(1.,0.,0.),1.:(0.,0.,0.)}}
    assert _station_balance_splits(geometry,targets,1.5)==((1,.25),)


@pytest.mark.parametrize('gradation',[1.,0.,float('inf'),True])
def test_invalid_gradation_refuses(gradation):
    with pytest.raises(MeshError):_station_balance_splits(NS(edges={}),{},gradation)


def test_actual_owner_union_preserves_requested_parameters_and_final_receipts():
    from test_local_curved_mesh import curved_loop_model
    from anymesher.native_v2 import NativeMeshingOptions
    from anymesher.refinement import SizeField
    model,face,edges,_=curved_loop_model();before=owner.to_dict(model)
    field=SizeField(model,.25,active_faces=(face,))
    groups=prepare_material_regions(model,(face,),{face:(face,)},order='linear',
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial'),
        supplied_seeding=False,edge_overrides={},geometry_approximation_error=.0025)
    bindings=tuple({id(b):b for b in groups.values()}.values())
    from anymesher.hybrid import _station_parameters
    requested={e:tuple(_station_parameters(model,e,2,field)) for e in edges}
    result=_prepare_curvilinear_station_union(model,bindings,{e:2 for e in edges},field,maximum_gradation=1.5)
    assert not _station_balance_splits(model,result,1.5)
    assert all(set(requested[e])<=set(result[e]) for e in edges)
    for binding in bindings:
        chords=binding._curvilinear_state['chords']
        owner.validate_prepared_material_boundary_chords_binding(model,binding.collection,chords)
        assert chords.chord_error==.0025
    assert owner.to_dict(model)==before
