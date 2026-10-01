"""Owner-evaluated refinement publishes only complete conforming material."""
from copy import deepcopy

import numpy as np
import pytest
from anygeometry.generators import cylinder
from anygeometry import to_dict

from anymesher import Mesh, mesh_to_dict
from anymesher._chart_triangle_refine import refine_triangle_face
from anymesher._cylindrical_chart import CylindricalMetricChart
from anymesher.errors import MeshError


@pytest.fixture(autouse=True)
def request_one_interior_refinement(monkeypatch):
    # Exercise staging on a square whose admissible interior diagonal can be
    # bisected safely. This checks the publication contract, not convergence
    # of a quality-repair algorithm; the application fixtures check the latter.
    import anymesher._chart_triangle_refine as implementation
    original=implementation._shape_failures
    def classify(coordinates,rows):
        return np.ones(len(rows),dtype=bool) if len(rows)==2 else original(coordinates,rows)
    monkeypatch.setattr(implementation,'_shape_failures',classify)


def case():
    owner = cylinder(10., 2., circumferential_segments=12)
    face = tuple(owner.group('shell'))[0].id
    chart = CylindricalMetricChart.from_geometry(owner, face)
    points = np.asarray(((.1,.1),(1.1,.1),(1.1,1.1),(.1,1.1)))
    xyz = chart.evaluate(points)
    mesh = Mesh(nodes={i+1:point.copy() for i,point in enumerate(xyz)},
                tris={1:(1,2,3),2:(1,3,4)},
                elements_of_face={face:[1,2]},
                elements_of_sheet={1:[1,2]},activity={1:True,2:True})
    return owner, face, chart, mesh


def run(owner, face, chart, mesh, cache, **options):
    return refine_triangle_face(mesh,owner,face,chart,
        protected_edges=((1,2),(2,3),(3,4),(4,1)),cache=cache,
        max_insertions=options.pop('max_insertions',100),
        max_work=options.pop('max_work',10000),**options)


def test_curved_interior_points_preserve_boundary_material_and_associations():
    owner,face,chart,mesh=case()
    source=to_dict(owner)
    before={node:point.tobytes() for node,point in mesh.nodes.items()}
    cache={}
    report=run(owner,face,chart,mesh,cache)
    assert report['insertions']>0
    assert report['added_elements']==2*report['insertions']
    assert not report['protected_coordinates_changed']
    assert all(mesh.nodes[node].tobytes()==raw for node,raw in before.items())
    assert set(mesh.elements_of_sheet[1])==set(mesh.tris)
    assert set(mesh.activity)==set(mesh.tris)
    assert face in cache
    nodes=np.asarray(list(mesh.nodes.values()))
    np.testing.assert_allclose(np.linalg.norm(nodes[:,:2],axis=1),10.,rtol=0.,atol=2e-14)
    assert to_dict(owner)==source
    from anymesher._chart_triangle_refine import _shape_failures
    ids=sorted(mesh.nodes); local={node:index for index,node in enumerate(ids)}
    assert not _shape_failures(np.asarray([mesh.nodes[node] for node in ids]),
        np.asarray([[local[node] for node in nodes] for nodes in mesh.tris.values()])).any()
    # The developed cylinder chart gives an independent exact area check.
    uv=owner.face_local_uv_many(face,np.asarray([mesh.nodes[node] for node in ids]))
    xy=uv*np.asarray((chart.circumferential_length,chart.axial_length))
    def area_of(a,b,c):
        first,second=xy[local[b]]-xy[local[a]],xy[local[c]]-xy[local[a]]
        return abs(first[0]*second[1]-first[1]*second[0])/2
    area=sum(area_of(a,b,c) for a,b,c in mesh.tris.values())
    assert area==pytest.approx(1.,rel=1e-12,abs=0.)


@pytest.mark.parametrize('options',({'max_insertions':0},{'max_work':0}))
def test_budget_exhaustion_leaves_mesh_owner_and_cache_exact(options):
    owner,face,chart,mesh=case()
    before=mesh_to_dict(mesh); source=to_dict(owner); cache={91:{(1,2):{4}}}
    initial=deepcopy(cache)
    with pytest.raises(MeshError,match='budget exhausted'):
        run(owner,face,chart,mesh,cache,**options)
    assert mesh_to_dict(mesh)==before and cache==initial and to_dict(owner)==source


def test_cancellation_after_staging_leaves_mesh_owner_and_cache_exact():
    owner,face,chart,mesh=case()
    before=mesh_to_dict(mesh); source=to_dict(owner); cache={}
    def cancel(stage):
        if stage.endswith('before publish'):
            raise RuntimeError('cancelled staged material refinement')
    with pytest.raises(RuntimeError,match='cancelled staged'):
        run(owner,face,chart,mesh,cache,cancellation_check=cancel)
    assert mesh_to_dict(mesh)==before and cache=={} and to_dict(owner)==source


def test_invalid_flip_never_publishes_staged_nodes(monkeypatch):
    from types import SimpleNamespace
    import anymesher._chart_triangle_refine as implementation
    owner,face,chart,mesh=case()
    before=mesh_to_dict(mesh); cache={}
    monkeypatch.setattr(implementation,'local_edge_flip',lambda points,rows,**kwargs:
        SimpleNamespace(triangles=np.zeros_like(rows),queue_visits=1))
    with pytest.raises(MeshError,match='invalid topology'):
        run(owner,face,chart,mesh,cache)
    assert mesh_to_dict(mesh)==before and cache=={}
