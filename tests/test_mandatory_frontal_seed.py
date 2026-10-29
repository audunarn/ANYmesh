"""Boundary-only setup is numerical seeding, never source or final mesh authority."""
from importlib import import_module
import numpy as np
import pytest
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.errors import MeshError
surface=import_module('anymesher.surface_mesh')


def inputs():
    outer=np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    explicit=np.array(((.5,.5),))
    constraints=[np.array(((.25,.25),(.75,.25)))]
    rows=np.vstack((outer,explicit,*constraints))
    return outer,explicit,constraints,rows,{i:100+i for i in range(len(rows))}


def test_actual_empty_lattice_path_retains_every_mandatory_row_and_id(monkeypatch):
    outer,explicit,constraints,rows,ids=inputs();before=rows.copy()
    def forbidden(*args,**kwargs): raise AssertionError('midpoint generation')
    monkeypatch.setattr(surface,'_refinement_midpoints',forbidden)
    settings=surface.SurfaceMeshOptions(target_size=.2,backend='python',
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial'))
    result=surface._run_quality_path('staggered_chart',outer,(),constraints,
        explicit,np.empty((0,2)),settings,None,allow_refinement=False,
        protected_node_ids=ids,protect_generated_seed_disks=True)
    actual=result['triangulation']
    mapping=dict(actual.protected_node_rows)
    assert set(mapping)==set(ids.values())
    for index,node in ids.items():
        np.testing.assert_array_equal(actual.points[mapping[node]],rows[index])
    assert len(actual.points)==len(rows)
    assert result['attempted_rounds']==result['attempted_added_points']==0
    np.testing.assert_array_equal(rows,before)


def test_empty_lattice_cancellation_preserves_mandatory_inputs():
    outer,explicit,constraints,rows,ids=inputs();before=rows.copy()
    error=RuntimeError('cancel mandatory setup')
    def cancel(phase): raise error
    with pytest.raises(RuntimeError) as caught:
        surface._run_quality_path('staggered_chart',outer,(),constraints,
            explicit,np.empty((0,2)),surface.SurfaceMeshOptions(backend='python'),
            cancel,allow_refinement=False,protected_node_ids=ids)
    assert caught.value is error
    np.testing.assert_array_equal(np.vstack((outer,explicit,*constraints)),before)


@pytest.mark.parametrize('value',[True,1,'mandatory'])
def test_private_seed_flag_cannot_bypass_owner_route(value):
    outer,explicit,constraints,rows,ids=inputs()
    with pytest.raises(MeshError,match='mandatory-only'):
        surface.mesh_planar_surface(outer,interior_points=explicit,
            constraints=constraints,target_size=.2,backend='python',
            _mandatory_only_frontal_seed=value,_boundary_is_seeded=True,
            _protected_node_ids=ids,native_options=NativeMeshingOptions(
                point_placement='frontal_delaunay',metric_mode='isotropic_spatial'))
