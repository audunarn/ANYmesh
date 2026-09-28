"""Independent published-mesh checks and physical measurements for SG1."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from anygeometry import to_dict
from anymesher.serialize import mesh_to_dict, mesh_from_dict
from anymesher.quad.high_order import evaluate_mapping, certify_mapping_validity, shape_values, shape_gradients
from .contract import LIMITS


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',',':'))


def geometry_signature(fixture):
    return hashlib.sha256(canonical(to_dict(fixture.geometry)).encode()).hexdigest()


def mesh_signature(mesh):
    data=mesh_to_dict(mesh)
    # Timings are observations, not deterministic mesh semantics.
    data.pop('hybrid_diagnostics', None)
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def check(checks, name, passed, **values):
    checks.append(dict(name=name,status='passed' if bool(passed) else 'failed',**values))


def quadrature(tri=False, order=8):
    x,w=np.polynomial.legendre.leggauss(order)
    if not tri:
        return np.array([(a,b) for a in x for b in x]),np.array([a*b for a in w for b in w])
    x,w=(x+1)/2,w/2
    return (np.array([(a,(1-a)*b) for a in x for b in x]),
            np.array([wa*wb*(1-a) for a,wa in zip(x,w) for wb in w]))


def owner_area(patch):
    previous=None
    for order in (8,16,32,64):
        p,w=quadrature(order=order)
        area=0.
        for (x,y),weight in zip(p,w):
            du,dv=patch.derivatives((x+1)/2,(y+1)/2)
            area+=weight*np.linalg.norm(np.cross(du,dv))/4
        if previous is not None and abs(area-previous)/area<=LIMITS['owner_area_convergence']:
            return float(area),order
        previous=area
    raise ValueError('owner area quadrature did not converge')


def samples(tri=False):
    if tri:
        return np.array([(i/8,j/8) for i in range(9) for j in range(9-i)])
    return np.array([(a,b) for a in np.linspace(-1,1,9) for b in np.linspace(-1,1,9)])


def physical_edges(xyz, ncorner, field):
    """Quadratic Lagrange edge integration; linear edges use their true chord."""
    x,w=np.polynomial.legendre.leggauss(8)
    records=[]
    for i in range(ncorner):
        a,b=xyz[i],xyz[(i+1)%ncorner]
        m=xyz[ncorner+i] if len(xyz)>ncorner else (a+b)/2
        pos=(.5*x*(x-1))[:,None]*a+(1-x*x)[:,None]*m+(.5*x*(x+1))[:,None]*b
        tangent=(x-.5)[:,None]*a-2*x[:,None]*m+(x+.5)[:,None]*b
        ds=np.linalg.norm(tangent,axis=1)
        sizes=np.asarray(field.size_at(pos)).reshape(-1)
        records.append((float(w@ds),float(w@(ds/sizes)),m))
    return records


def audit(fixture,mesh,h,graded):
    checks=[]
    cells={**mesh.quads,**mesh.tris}
    field=fixture.field(h,graded)
    diameter=max(1.,float(np.linalg.norm(np.ptp(np.array(list(mesh.nodes.values())),axis=0))))
    support_tol=LIMITS['support_relative']*diameter
    # CH11 original controls have an absolute 1e-10 support limit.
    if fixture.case.startswith('T-'):
        support_tol=min(support_tol,1e-10)
    all_edges=defaultdict(list)
    face_edges={}
    max_support=max_geometry=max_scaled=max_normal=max_aspect=0.
    min_quality=1.
    total_area=total_owner=quad_area=0.
    ratios=[]; lengths=[]; core_lengths=[]; remote_lengths=[]
    geometry_by_cell={}; quality_by_cell={}; corner_nodes=set()
    certificate_counts=defaultdict(int)
    face_records={}
    owned_shells=[]
    for face,patch in fixture.patches.items():
        polys=[]; incidence=defaultdict(list)
        element_ids=mesh.elements_of_face.get(face,())
        owned_shells.extend(element_ids)
        check(checks,f'face/{face}/nonempty',bool(element_ids))
        for eid in element_ids:
            body=cells[eid]; nc=4 if eid in mesh.quads else 3
            family=('Q' if nc==4 else 'T')+str(len(body))
            xyz=np.array([mesh.nodes[n] for n in body]); corner_nodes.update(body[:nc])
            uvs=np.array([patch.uv(p) for p in xyz])
            polys.append(Polygon(uvs[:nc]))
            support=max(np.linalg.norm(patch.point(*uv)-p) for uv,p in zip(uvs,xyz))
            max_support=max(max_support,float(support))
            normal=patch.normal(*np.mean(uvs[:nc],axis=0))
            cert=certify_mapping_validity(xyz,family,reference_normal=normal)
            certificate_counts[cert.status.value]+=1
            ev=evaluate_mapping(xyz,family,samples(nc==3),reference_normal=normal)
            min_quality=min(min_quality,float(np.min(ev.normalized_quality)))
            quality_by_cell[eid]=float(np.min(ev.normalized_quality))
            cell_error=0.
            for point,nvec in zip(ev.points,ev.jacobian_vector):
                uv=patch.uv(point)
                residual=float(np.linalg.norm(point-patch.point(*uv)))
                cell_error=max(cell_error,residual)
                max_scaled=max(max_scaled,residual/float(np.asarray(field.size_at(point)).reshape(-1)[0]))
                dot=np.dot(nvec/np.linalg.norm(nvec),patch.normal(*uv))
                max_normal=max(max_normal,float(np.degrees(np.arccos(np.clip(dot,-1.,1.)))))
            geometry_by_cell[eid]=cell_error; max_geometry=max(max_geometry,cell_error)
            p,w=quadrature(nc==3)
            area=float(w@evaluate_mapping(xyz,family,p,reference_normal=normal).jacobian_magnitude)
            total_area+=area
            if nc==4: quad_area+=area
            edges=physical_edges(xyz,nc,field)
            max_aspect=max(max_aspect,max(e[0] for e in edges)/min(e[0] for e in edges))
            for i,(length,ratio,mid) in enumerate(edges):
                a,b=body[i],body[(i+1)%nc]
                key=tuple(sorted((a,b)))
                midside=body[nc+i] if len(body)>nc else None
                incidence[key].append((a,b,midside,eid))
                if not all_edges[key]:
                    ratios.append(ratio); lengths.append(length)
                    distance=float(np.linalg.norm(mid-fixture.center))
                    if distance<=.35: core_lengths.append(length)
                    if distance>=.7: remote_lengths.append(length)
                all_edges[key].append((a,b,midside,eid,nc,face))
        face_edges[face]=incidence
        if polys and all(p.is_valid for p in polys):
            union=unary_union(polys)
            overlap=sum(p.area for p in polys)-union.area
            gap=union.symmetric_difference(box(0,0,1,1)).area
            check(checks,f'face/{face}/coverage',overlap<=1e-9 and gap<=1e-9,overlap=overlap,gap=gap)
        else:
            check(checks,f'face/{face}/coverage',False,reason='invalid or empty UV polygons')
        expected=set()
        for edge,forward in fixture.face_edges[face]:
            chain=mesh.nodes_of_edge.get(edge,())
            check(checks,f'face/{face}/edge/{edge}/chain',len(chain)>=2)
            step=2 if mesh.order=='quadratic' else 1
            for i in range(0,len(chain)-step,step):
                a,b=chain[i],chain[i+step]; key=tuple(sorted((a,b))); expected.add(key)
                entries=incidence.get(key,())
                want=(a,b) if forward else (b,a)
                ok=len(entries)==1 and entries[0][:2]==want
                if step==2: ok=ok and entries[0][2]==chain[i+1] if entries else False
                check(checks,f'face/{face}/edge/{edge}/segment/{i}',ok)
        boundary={k for k,v in incidence.items() if len(v)==1}
        check(checks,f'face/{face}/boundary',boundary==expected)
        check(checks,f'face/{face}/incidence',all(len(v)==1 or (len(v)==2 and v[0][:2]==v[1][1::-1]) for v in incidence.values()))
        own,order=owner_area(patch); total_owner+=own
        face_records[face]=dict(owner_area=own,owner_quadrature_order=order,elements=len(element_ids))
    mixed=0
    check(checks,'shell/complete-ownership',
          len(owned_shells)==len(set(owned_shells)) and set(owned_shells)==set(cells))
    if fixture.sheet is not None:
        check(checks,'sheet/complete-ownership',
              set(mesh.elements_of_sheet)=={fixture.sheet}
              and len(mesh.elements_of_sheet[fixture.sheet])==len(cells)
              and set(mesh.elements_of_sheet[fixture.sheet])==set(cells))
    for key,entries in all_edges.items():
        check(checks,f'global-edge/{key}/incidence',len(entries)<=2)
        if len(entries)==2:
            check(checks,f'global-edge/{key}/midside',entries[0][2]==entries[1][2])
            if entries[0][4]!=entries[1][4]: mixed+=1
    if fixture.case.startswith('T-'):
        check(checks,'mixed-Q8-T6-interface',bool(mesh.tris) and mixed>0,mixed_edges=mixed)
    owned_beams=[]
    for mid in fixture.members:
        bodies=[mesh.beams[e] for e in mesh.elements_of_member.get(mid,())]
        owned_beams.extend(mesh.elements_of_member.get(mid,()))
        check(checks,f'member/{mid}/ownership',bool(bodies))
        check(checks,f'member/{mid}/station-chain',set(mesh.nodes_of_member.get(mid,()))==set(n for b in bodies for n in b))
        chain=list(mesh.nodes_of_edge.get(fixture.member_edge,()))
        entity=fixture.geometry.edges[fixture.member_edge]
        start=fixture.geometry.vertex_position(entity.start); end=fixture.geometry.vertex_position(entity.end)
        direction=end-start; length2=float(direction@direction)
        stations=[float((mesh.nodes[n]-start)@direction/length2) for n in chain]
        check(checks,f'member/{mid}/ordered-stations',len(chain)>=3 and len(set(chain))==len(chain)
              and abs(stations[0])<=1e-10 and abs(stations[-1]-1)<=1e-10
              and all(a<b for a,b in zip(stations,stations[1:])))
        check(checks,f'member/{mid}/source-line',all(np.linalg.norm(mesh.nodes[n]-start-t*direction)<=support_tol for n,t in zip(chain,stations)))
        check(checks,f'member/{mid}/endpoints',bool(chain) and chain[0]==mesh.node_of_vertex[entity.start] and chain[-1]==mesh.node_of_vertex[entity.end])
        step=2 if mesh.order=='quadratic' else 1
        expected=[tuple(chain[i:i+step+1]) for i in range(0,len(chain)-step,step)]
        check(checks,f'member/{mid}/ordered-bodies',bodies==expected and len(chain)==step*len(bodies)+1)
        check(checks,f'member/{mid}/ordered-owner-chain',list(mesh.nodes_of_member.get(mid,()))==chain)
        for body in bodies:
            if mesh.order=='quadratic':
                check(checks,f'B3/{body}/midpoint',len(body)==3 and np.linalg.norm(mesh.nodes[body[1]]-(mesh.nodes[body[0]]+mesh.nodes[body[2]])/2)<=support_tol)
        check(checks,'attachment/count',len(mesh.couplings)==1)
    check(checks,'beam/complete-ownership',len(owned_beams)==len(set(owned_beams)) and set(owned_beams)==set(mesh.beams))
    for eid,c in mesh.couplings.items():
        declared_member_nodes={node for member in fixture.members
                               for body_id in mesh.elements_of_member.get(member,())
                               for node in mesh.beams.get(body_id,())}
        check(checks,f'coupling/{eid}/beam-owner',c.beam_node in declared_member_nodes)
        coords=np.array([mesh.nodes[n] for n in c.plate_nodes]); weights=np.asarray(c.weights)
        projected=weights@coords; beam=mesh.nodes[c.beam_node]
        check(checks,f'coupling/{eid}/eccentricity',np.linalg.norm(beam-projected-np.asarray(c.eccentricity))<=support_tol)
        check(checks,f'coupling/{eid}/partition',abs(sum(c.weights)-1)<=1e-10)
        # Fit to the DECLARED owner point, never to a point defined by the
        # coupling's own weights. Thus consistent-but-displaced records fail.
        from scipy.optimize import least_squares
        face=fixture.faces[0]; source=fixture.patches[face].point(.5,.5)
        check(checks,f'coupling/{eid}/source-station',np.linalg.norm(beam-source)<=support_tol)
        check(checks,f'coupling/{eid}/owner',any(tuple(c.plate_nodes)==tuple(cells[e]) for e in mesh.elements_of_face[face]))
        if len(c.plate_nodes) in (8,6,4,3):
            fam=('Q' if len(c.plate_nodes) in (4,8) else 'T')+str(len(c.plate_nodes))
            if fam[0]=='Q':
                # Start the independent physical closest-point solve at the
                # parametric position recoverable from the record. Near an
                # element edge, a centre start can terminate a few 1e-10 m
                # away from the same constrained optimum.
                reference=np.asarray(((-1,-1),(1,-1),(1,1),(-1,1),
                                      (0,-1),(1,0),(0,1),(-1,0))[:len(weights)])
                seed=weights@reference
            else:
                seed=[1/3,1/3]
            fit=least_squares(lambda p:shape_values(fam,p)@coords-source,seed,
                              jac=lambda p:coords.T@shape_gradients(fam,p),
                              gtol=1e-14,ftol=1e-14,xtol=1e-14)
            inside=(max(abs(fit.x))<=1+1e-8 if fam[0]=='Q' else min(fit.x)>=-1e-8 and sum(fit.x)<=1+1e-8)
            check(checks,f'coupling/{eid}/parametric-domain',inside)
            check(checks,f'coupling/{eid}/weights',np.max(np.abs(shape_values(fam,fit.x)-weights))<=1e-8)
            check(checks,f'coupling/{eid}/declared-projection',np.linalg.norm(projected-shape_values(fam,fit.x)@coords)<=support_tol)
        else: check(checks,f'coupling/{eid}/supported-master-cell',False)
    if mesh.order=='quadratic':
        check(checks,'certified-positive',certificate_counts.get('CERTIFIED_POSITIVE',0)==len(cells),counts=dict(certificate_counts))
        check(checks,'support',max_support<=support_tol,value=max_support,limit=support_tol)
        check(checks,'geometry-over-h',max_scaled<=LIMITS['geometry_over_h'],value=max_scaled)
        check(checks,'normal-degrees',max_normal<=LIMITS['normal_degrees'],value=max_normal)
        check(checks,'area-relative',abs(total_area-total_owner)/total_owner<=LIMITS['area_relative'],value=abs(total_area-total_owner)/total_owner)
        check(checks,'normalized-jacobian',min_quality>=LIMITS['normalized_jacobian'],value=min_quality)
        check(checks,'aspect',max_aspect<=LIMITS['aspect'],value=max_aspect)
        median=float(np.median(ratios)); fraction=float(np.mean((np.array(ratios)>=.25)&(np.array(ratios)<=2)))
        check(checks,'sizing-median',.5<=median<=1.5,value=median)
        check(checks,'sizing-fraction',fraction>=.95,value=fraction)
        check(checks,'sizing-max',max(ratios)<=4,value=max(ratios))
    check(checks,'serialization',mesh_signature(mesh_from_dict(mesh_to_dict(mesh)))==mesh_signature(mesh))
    return dict(checks=checks,counts=dict(nodes=len(mesh.nodes),quads=len(mesh.quads),triangles=len(mesh.tris),beams=len(mesh.beams),couplings=len(mesh.couplings),equivalent=len(mesh.quads)+len(mesh.tris)/2),
        faces=face_records,quad_fraction_count=len(mesh.quads)/len(cells),quad_fraction_area=quad_area/total_area,
        max_geometry_error=max_geometry,max_support=max_support,support_tolerance=support_tol,
        max_geometry_over_h=max_scaled,max_normal_degrees=max_normal,min_normalized_jacobian=min_quality,
        max_aspect=max_aspect,physical_area=total_area,owner_area=total_owner,
        size_ratios=ratios,physical_edge_lengths=lengths,core_corner_count=int(sum(np.linalg.norm(mesh.nodes[n]-fixture.center)<=.35 for n in corner_nodes)),
        core_median=float(np.median(core_lengths)) if core_lengths else None,
        remote_median=float(np.median(remote_lengths)) if remote_lengths else None,
        geometry_by_cell=geometry_by_cell,quality_by_cell=quality_by_cell)


def promotion_checks(linear,quadratic):
    checks=[]
    check(checks,'promotion/nodes',all(n in quadratic.nodes and np.array_equal(p,quadratic.nodes[n]) for n,p in linear.nodes.items()))
    check(checks,'promotion/quads',{e:tuple(b[:4]) for e,b in quadratic.quads.items()}==linear.quads)
    check(checks,'promotion/triangles',{e:tuple(b[:3]) for e,b in quadratic.tris.items()}==linear.tris)
    for key in ('elements_of_face','elements_of_sheet','elements_of_member'):
        check(checks,'promotion/'+key,getattr(linear,key)==getattr(quadratic,key))
    for edge,chain in linear.nodes_of_edge.items():
        check(checks,f'promotion/stations/{edge}',quadratic.nodes_of_edge.get(edge,[])[::2]==chain)
    return checks
