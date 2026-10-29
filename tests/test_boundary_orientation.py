"""Public owner-loop winding contract, independent of support-chart handedness."""
from copy import deepcopy
from collections import defaultdict
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np
import pytest

from anygeometry import GeometryModel, OrientedEdge, Plane, to_dict
from anygeometry.certified_hydrostatics import CertifiedHydrostaticsRefusal
from anymesher import (
    BoundaryOrientationError, Mesh, orient_displacement_boundary,
    generate_displacement_boundary_mesh_result, mesh_from_dict, mesh_to_dict,
)


def box(shift=(0, 0, 0)):
    g = GeometryModel()
    points = [np.array((x, y, z), dtype=float)+shift
              for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    vertices = [g.add_point(*p) for p in points]
    loops = ((0,2,3,1), (4,5,7,6), (0,4,6,2),
             (1,3,7,5), (0,1,5,4), (2,6,7,3))
    edges, faces = {}, []
    for i, loop in enumerate(loops):
        uses = []
        for a, b in zip(loop, loop[1:]+loop[:1]):
            key = frozenset((a,b))
            if key not in edges:
                edges[key] = g.add_line(vertices[a], vertices[b])
            eid = edges[key]
            uses.append(OrientedEdge(eid, g.edges[eid].start == vertices[a]))
        origin = points[loop[0]]
        u, v = points[loop[1]]-origin, points[loop[2]]-origin
        # Every other support chart has opposite handedness to source loops.
        if i % 2:
            u, v = v, u
        faces.append(g.add_face_from_loop(uses, surface=Plane(origin, u, v)))
    m = Mesh(geometry_model_id=g.model_id, geometry_revision=g.revision)
    m.node_of_vertex = {v:101+7*i for i,v in enumerate(vertices)}
    m.nodes = {m.node_of_vertex[v]:p.copy() for v,p in zip(vertices,points)}
    m.nodes_of_edge = {e:[m.node_of_vertex[edge.start],m.node_of_vertex[edge.end]]
                       for e,edge in g.edges.items()}
    for i, (f, loop) in enumerate(zip(faces,loops)):
        a,b,c,d = (m.node_of_vertex[vertices[j]] for j in loop)
        e = 10+i*5
        m.tris[e],m.tris[e+1] = (a,b,c),(a,c,d)
        m.elements_of_face[f] = [e,e+1]
        if i % 2:
            for eid in (e,e+1):
                x,y,z=m.tris[eid];m.tris[eid]=(x,z,y)
    return g,m,faces


def opposite(mesh):
    incidence=defaultdict(list)
    for tri in mesh.tris.values():
        for a,b in zip(tri,tri[1:]+tri[:1]):
            incidence[tuple(sorted((a,b)))].append((a,b))
    return all(len(v)==2 and v[0]==v[1][::-1] for v in incidence.values())


@pytest.mark.parametrize('shift',[(0,0,0),(200,-700,300)])
def test_public_orients_source_loops_preserves_input_and_associations(shift):
    g,m,faces=box(shift)
    before,source=mesh_to_dict(m),to_dict(g)
    assert not opposite(m)
    r=orient_displacement_boundary(m,g,face_ids=iter(faces))
    assert opposite(r.mesh)
    assert mesh_to_dict(m)==before==mesh_to_dict(r.original_mesh)
    assert to_dict(g)==source
    output=mesh_to_dict(r.mesh)
    for key in before:
        if key not in ('tris','hybrid_diagnostics'):
            assert output[key]==before[key],key
    assert {v['outward_chart_sign'] for v in r.evidence['source_face_support'].values()}=={-1,1}
    for tri in r.mesh.tris.values():
        xyz=np.array([r.mesh.nodes[n] for n in tri])
        normal=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
        assert normal@(np.mean(xyz,axis=0)-np.array(shift)-.5)>0
    assert mesh_to_dict(mesh_from_dict(output))==output
    again=orient_displacement_boundary(m,g,face_ids=reversed(faces))
    assert mesh_to_dict(again.mesh)==output


@pytest.mark.parametrize('defect,category',[
    ('open','TOPOLOGY'),('duplicate','TOPOLOGY'),('station','ASSOCIATION'),
    ('order','ASSOCIATION'),('face','ASSOCIATION'),('extra_owner','ASSOCIATION'),
    ('off_source','SOURCE_SUPPORT'),('unused','TOPOLOGY'),('nan','GEOMETRY'),
    ('quadratic','UNSUPPORTED'),('beam','UNSUPPORTED'),('revision','SOURCE_BINDING'),
])
def test_refusal_is_atomic(defect,category):
    g,m,faces=box()
    if defect=='open':
        e=min(m.tris);del m.tris[e];m.elements_of_face[faces[0]].remove(e)
    elif defect=='duplicate':
        m.tris[999]=m.tris[min(m.tris)];m.elements_of_face[faces[0]].append(999)
    elif defect=='station':del m.nodes_of_edge[min(m.nodes_of_edge)]
    elif defect=='order':m.nodes_of_edge[min(m.nodes_of_edge)].reverse()
    elif defect=='face':m.elements_of_face[faces[0]].pop()
    elif defect=='extra_owner':m.elements_of_face[faces[1]].append(min(m.tris))
    elif defect=='off_source':m.nodes[min(m.nodes)]+=np.array([.01,.02,.03])
    elif defect=='unused':m.nodes[999]=np.zeros(3)
    elif defect=='nan':m.nodes[min(m.nodes)][0]=np.nan
    elif defect=='quadratic':m.order='quadratic'
    elif defect=='beam':m.beams[999]=(min(m.nodes),max(m.nodes))
    elif defect=='revision':m.geometry_revision-=1
    before=deepcopy(m)
    with pytest.raises(BoundaryOrientationError) as exc:
        orient_displacement_boundary(m,g,face_ids=faces)
    assert exc.value.category==category
    assert m.tris==before.tris
    for n in m.nodes:np.testing.assert_equal(m.nodes[n],before.nodes[n])


def test_wrong_model_and_duplicate_scope():
    g,m,faces=box();other,_,_=box()
    with pytest.raises(BoundaryOrientationError,match='identity'):
        orient_displacement_boundary(m,other,face_ids=faces)
    with pytest.raises(BoundaryOrientationError,match='unique'):
        orient_displacement_boundary(m,g,face_ids=faces+faces[:1])


def test_late_cancellation_and_stale_source(monkeypatch):
    g,m,faces=box();before=mesh_to_dict(m)
    original=g.face_normal_many
    cancelled=False
    def normals(*args):
        nonlocal cancelled
        result=original(*args);cancelled=True
        return result
    monkeypatch.setattr(g,'face_normal_many',normals)
    with pytest.raises(BoundaryOrientationError) as exc:
        orient_displacement_boundary(m,g,face_ids=faces,cancellation_check=lambda:cancelled)
    assert exc.value.category=='CANCELLED' and mesh_to_dict(m)==before
    changed=False
    def changed_normals(*args):
        nonlocal changed
        result=original(*args)
        if not changed:g.add_point(5,5,5);changed=True
        return result
    monkeypatch.setattr(g,'face_normal_many',changed_normals)
    with pytest.raises(CertifiedHydrostaticsRefusal):
        orient_displacement_boundary(m,g,face_ids=faces)
    assert mesh_to_dict(m)==before


def test_generation_wrapper_uses_frozen_request_and_keeps_raw_mesh(monkeypatch):
    import anymesher.hybrid as hybrid
    g,m,faces=box();seen={}
    # Native generation has ephemeral, unpickleable owner/cache state. Durable
    # public serialization is the cloning contract, not deepcopy(Mesh).
    m.ephemeral_owner_cache=MappingProxyType({'owner':object()})
    def generate(model,**options):
        assert model is g
        seen.update(options)
        options['cancellation_check']('hybrid generation start')
        return SimpleNamespace(mesh=m)
    monkeypatch.setattr(hybrid,'generate_hybrid_mesh_result',generate)
    r=generate_displacement_boundary_mesh_result(g,target_size=.2,face_ids=iter(faces))
    assert opposite(r.mesh) and not opposite(r.original_mesh)
    assert {k:seen[k] for k in ('strategy','order','recombine','structural_preparation','certification_mode','mutation_policy','member_ids','beam_edges')}==dict(strategy='native',order='linear',recombine=False,structural_preparation=False,certification_mode='strict',mutation_policy='read_only',member_ids=(),beam_edges=())


def test_saved_oriented_mesh_can_be_rechecked_with_verifiable_receipt():
    from anymesher.boundary_orientation import _digest
    g,m,faces=box()
    first=orient_displacement_boundary(m,g,face_ids=faces)
    second=orient_displacement_boundary(first.mesh,g,face_ids=faces)
    assert second.mesh.tris==first.mesh.tris
    assert not second.evidence['flipped_element_ids']
    stripped=mesh_from_dict(mesh_to_dict(second.mesh))
    del stripped.hybrid_diagnostics['boundary_orientation']
    assert _digest(stripped)==second.evidence['oriented_mesh_sha256_before_receipt']


@pytest.mark.parametrize('second_node,first_element,second_element',[(9,10,1406),(15,20,5182)])
def test_original_id_witness_pattern(second_node,first_element,second_element):
    """Synthetic shared-face regression using reported B1 witness IDs, not B1 evidence."""
    g,m,faces=box()
    # Source faces1/6 share geometry vertices3/4 in this box fixture.
    a=m.node_of_vertex[3];b=m.node_of_vertex[4]
    rename={a:2,b:second_node}
    m.nodes={rename.get(n,n):p for n,p in m.nodes.items()}
    m.node_of_vertex={v:rename.get(n,n) for v,n in m.node_of_vertex.items()}
    m.nodes_of_edge={e:[rename.get(n,n) for n in ns] for e,ns in m.nodes_of_edge.items()}
    m.tris={e:tuple(rename.get(n,n) for n in t) for e,t in m.tris.items()}
    old_first=m.elements_of_face[faces[0]][0]
    old_second=m.elements_of_face[faces[5]][1]
    remap={e:e+10000 for e in m.tris}
    remap.update({old_first:first_element,old_second:second_element})
    m.tris={remap.get(e,e):t for e,t in m.tris.items()}
    m.elements_of_face={f:[remap.get(e,e) for e in es] for f,es in m.elements_of_face.items()}
    edge=(2,second_node)
    for e in (first_element,second_element):
        t=m.tris[e]
        assert edge in list(zip(t,t[1:]+t[:1]))
    r=orient_displacement_boundary(m,g,face_ids=faces)
    assert opposite(r.mesh)


@pytest.mark.parametrize('pinch',[False,True])
def test_disconnected_or_pinched_closed_mesh_refused(pinch):
    g,m,faces=box()
    nodes={n:n+1000 for n in m.nodes}
    if pinch:nodes[min(m.nodes)]=min(m.nodes)
    for n,new in nodes.items():
        if new not in m.nodes:m.nodes[new]=m.nodes[n].copy()
    for f,es in list(m.elements_of_face.items()):
        for e in list(es):
            m.tris[e+1000]=tuple(nodes[n] for n in m.tris[e])
            es.append(e+1000)
    with pytest.raises(BoundaryOrientationError) as exc:
        orient_displacement_boundary(m,g,face_ids=faces)
    assert exc.value.category=='TOPOLOGY'
    assert ('link' if pinch else 'disconnected') in str(exc.value)


def test_nonorientable_closed_complex_refused():
    g,m,faces=box()
    triangles=((1,2,3),(1,2,4),(1,3,5),(1,4,6),(1,5,6),
               (2,3,6),(2,4,5),(2,5,6),(3,4,5),(3,4,6))
    m.nodes={n:np.array([n,n*n,n*n*n],dtype=float) for n in range(1,7)}
    m.tris={e:t for e,t in enumerate(triangles,1)}
    m.elements_of_face={f:[] for f in faces}
    for e in m.tris:m.elements_of_face[faces[(e-1)%6]].append(e)
    with pytest.raises(BoundaryOrientationError,match='not orientable'):
        orient_displacement_boundary(m,g,face_ids=faces)


def test_source_change_between_early_binding_and_prepare_refused(monkeypatch):
    import anygeometry.certified_hydrostatics as owner
    g,m,faces=box();original=owner.prepare_certified_hydrostatics
    def prepare(model,*args,**kwargs):
        model.add_point(8,8,8)
        return original(model,*args,**kwargs)
    monkeypatch.setattr(owner,'prepare_certified_hydrostatics',prepare)
    with pytest.raises(BoundaryOrientationError,match='between mesh binding'):
        orient_displacement_boundary(m,g,face_ids=faces)


@pytest.mark.parametrize('change',['source','input'])
def test_change_during_final_binding_refused(monkeypatch,change):
    import anygeometry.certified_hydrostatics as owner
    g,m,faces=box();original=owner.check_preparation
    def check(model,*args,**kwargs):
        result=original(model,*args,**kwargs)
        if change=='source':model.add_point(8,8,8)
        else:m.activity[101]=.5
        return result
    monkeypatch.setattr(owner,'check_preparation',check)
    with pytest.raises(BoundaryOrientationError) as exc:
        orient_displacement_boundary(m,g,face_ids=faces)
    assert exc.value.category==('SOURCE_BINDING' if change=='source' else 'INPUT_CHANGED')
