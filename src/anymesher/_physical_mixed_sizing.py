"""Bounded conforming mixed-cell midpoint proposals under published quality.

Only internal edges are cut. No centroid, source receipt or geometry truth is
constructed here. The caller owns the canonical evaluator and final admission.
"""
from collections import Counter
from dataclasses import replace
import numpy as np


def _edges(nodes):
    return tuple(tuple(sorted((int(a),int(b)))) for a,b in zip(nodes,nodes[1:]+nodes[:1]))


def _boundary(cells):
    result=Counter((int(a),int(b)) for nodes in cells for a,b in zip(nodes,nodes[1:]+nodes[:1]))
    for a,b in list(result):
        count=min(result[a,b],result[b,a]);result[a,b]-=count;result[b,a]-=count
    return +result


def _positive(points,nodes):
    from .native_v2 import orient2d
    if len(nodes)==3:return orient2d(*(points[int(n)] for n in nodes))>0
    return all(orient2d(points[nodes[i-1]],points[nodes[i]],points[nodes[(i+1)%4]])>0 for i in range(4))


def _eligible_chart(evaluate):
    from types import MethodType
    from ._analytic_metric_chart import AnalyticMetricChart,_REFERENCE_ANALYTIC_HELPERS,_REFERENCE_ANALYTIC_EVALUATE
    from ._material_region_binding import (MaterialRegionBinding,_material_chart_snapshot,
        _boundary_callable_pin,_CanonicalMaterialRead,_CANONICAL_MATERIAL_EVALUATE,_CANONICAL_MATERIAL_EVALUATOR_PIN)
    if type(evaluate) is not MethodType:return None
    chart=evaluate.__self__
    if (type(chart) is not AnalyticMetricChart or evaluate.__func__ is not _REFERENCE_ANALYTIC_EVALUATE[0]
            or evaluate.__func__.__code__ is not _REFERENCE_ANALYTIC_EVALUATE[1]
            or chart.check is not None or type(chart.region_binding) is not MaterialRegionBinding
            or any(name in vars(chart) for name,_ in _REFERENCE_ANALYTIC_HELPERS)
            or any(_boundary_callable_pin(vars(AnalyticMetricChart).get(name))!=pin for name,pin in _REFERENCE_ANALYTIC_HELPERS)
            or _CanonicalMaterialRead.evaluate is not _CANONICAL_MATERIAL_EVALUATE
            or _boundary_callable_pin(_CanonicalMaterialRead.evaluate)!=_CANONICAL_MATERIAL_EVALUATOR_PIN):return None
    chart._current()
    return chart if _material_chart_snapshot(chart,create=True) is not None else None


def _cover(report,protected):
    from .errors import MeshError
    mesh=report.mesh;byid={int(value):row for row,value in enumerate(mesh.triangle_ids)}
    qrows={int(mesh.quad_ids[row]):int(row) for row in np.flatnonzero(mesh.quad_active)}
    pairs,created,scores=report.paired_triangle_ids,report.created_quad_ids,report.scores
    if (any(type(a) is not np.ndarray for a in (pairs,created,scores))
            or pairs.shape!=(len(qrows),2) or created.shape!=(len(qrows),) or scores.shape!=(len(qrows),)
            or pairs.dtype.kind not in 'iu' or created.dtype.kind not in 'iu' or scores.dtype.kind not in 'fiu'
            or not np.isfinite(scores).all() or set(map(int,created))!=set(qrows)):
        raise MeshError('mixed sizing requires current complete quadrilateral pair metadata')
    covers={};claimed=set()
    for qid,pair in zip(report.created_quad_ids,report.paired_triangle_ids):
        if len(pair)!=2 or any(int(identifier) not in byid for identifier in pair):raise MeshError('mixed sizing pair identifiers are invalid')
        rows=tuple(byid[int(identifier)] for identifier in pair)
        cells=[tuple(map(int,mesh.triangle_connectivity[row])) for row in rows]
        quad=tuple(map(int,mesh.quad_connectivity[qrows[int(qid)]]))
        shared=set(_edges(cells[0]))&set(_edges(cells[1]))
        if (len(set(rows))!=2 or claimed.intersection(rows) or any(mesh.triangle_active[row] for row in rows)
                or len(shared)!=1 or shared&protected or _boundary(cells)!=_boundary([quad])):
            raise MeshError('mixed sizing requires a nonoverlapping unprotected T3 cover')
        if any(int(mesh.triangle_owner_handles[row])!=int(mesh.quad_owner_handles[qrows[int(qid)]]) for row in rows):
            raise MeshError('mixed sizing cover changes material owner')
        covers[int(qid)]=rows;claimed.update(rows)
    if set(covers)!=set(qrows) or claimed!=set(map(int,np.flatnonzero(~mesh.triangle_active))):raise MeshError('mixed sizing has uncovered inactive triangles')
    return covers


def _frontier(mesh,limit):
    records=[];incidence={}
    for kind,cells,active,ids,owners in (('T',mesh.triangle_connectivity,mesh.triangle_active,mesh.triangle_ids,mesh.triangle_owner_handles),('Q',mesh.quad_connectivity,mesh.quad_active,mesh.quad_ids,mesh.quad_owner_handles)):
        rows=np.flatnonzero(active)
        # Batched arithmetic retains each cell's edge and reduction order.
        # Records/incidence are rebuilt from the current committed topology.
        xyz=mesh.node_coordinates[cells[rows]]
        lengths=np.linalg.norm(np.roll(xyz,-1,axis=1)-xyz,axis=2).mean(axis=1)
        for row,length in zip(rows,lengths):
            nodes=tuple(map(int,cells[row]));length=float(length)
            record=(kind,int(row),int(ids[row]),nodes,int(owners[row]),length)
            index=len(records);records.append(record)
            for edge in _edges(nodes):incidence.setdefault(edge,[]).append(index)
    frontier=[]
    for edge,attached in incidence.items():
        if len(attached)!=2:continue
        a,b=(records[i] for i in attached)
        if min(a[5],b[5])<=0:continue
        ratio=max(a[5]/b[5],b[5]/a[5])
        if ratio>limit:
            larger=attached[0] if a[5]>=b[5] else attached[1]
            frontier.append((-ratio,records[larger][2],edge,larger))
    frontier.sort()
    return records,incidence,frontier


def _key(children,xyz,settings):
    from .quality_v2 import quad_candidate_quality,triangle_quality
    counts=0;jac=[]
    for kind,nodes in children:
        if kind=='Q':m=quad_candidate_quality(xyz,nodes)
        else:
            q=triangle_quality(xyz,np.asarray([nodes],dtype=np.int64))
            m={name:float(getattr(q,name)[0]) for name in ('scaled_jacobian','aspect_ratio','minimum_angle','maximum_angle','warpage')}
        counts+=sum((m['scaled_jacobian']<settings.min_scaled_jacobian,m['aspect_ratio']>settings.max_aspect_ratio,m['minimum_angle']<settings.min_angle,m['maximum_angle']>settings.max_angle,m['warpage']>settings.max_warpage))
        jac.append(m['scaled_jacobian'])
    return counts,-min(jac)


def _children(parent,mids,points,xyz,settings,charge,outside=None):
    kind,_,_,nodes,_,_=parent;split=[i for i,e in enumerate(_edges(nodes)) if e in mids]
    if len(split)==1:
        i=split[0];r=nodes[i:]+nodes[:i];m=mids[tuple(sorted((r[0],r[1]))) ]
        options=([[('T',(r[0],m,r[2])),('T',(m,r[1],r[2]))]] if kind=='T' else
            [[('T',(r[0],m,r[3])),('Q',(m,r[1],r[2],r[3]))],[('Q',(r[0],m,r[2],r[3])),('T',(m,r[1],r[2]))]])
    elif kind=='Q' and len(split)==2 and (split[1]-split[0])%4==2:
        i=split[0];r=nodes[i:]+nodes[:i];m=mids[tuple(sorted((r[0],r[1])))];n=mids[tuple(sorted((r[2],r[3])))]
        options=[[('Q',(r[0],m,n,r[3])),('Q',(m,r[1],r[2],n))]]
    else:return None
    viable=[]
    for option in options:
        charge()
        if all(_positive(points,child) for _,child in option):viable.append((_key(option,xyz,settings),option))
    if outside is not None and kind=='Q' and len(split)==1:
        clear=[value for value in viable if value[0][0]==0]
        if len(clear)==2:
            # Two shape-clear (T,Q) options: lower local growth excess against the external neighbours first.
            return min(clear,key=lambda value:(_star_objective([value[1]],xyz,settings,outside)[1],value[0]))[1]
    return min(viable,key=lambda value:value[0])[1] if viable else None


def _neighbour_sizes(parents,records,incidence):
    ids={p[2] for p in parents};outside={}
    for p in parents:
        for edge in _edges(p[3]):
            for i in incidence[edge]:
                if records[i][2] not in ids:outside[edge]=records[i][5]
    return outside


def _search_order(objective):
    # Shape-clear candidates rank by growth excess first; violating ones are steered by slack toward feasibility.
    return objective if objective[0]==0 else (objective[0],objective[2],objective[1])


def _star_objective(children,xyz,settings,outside):
    """Lexicographic local score: (shape violations, growth excess, -minimum normalised slack)."""
    from .quality_v2 import quad_candidate_quality,triangle_quality
    count=0;slack=[];lengths={}
    for group in children:
        for kind,nodes in group:
            if kind=='Q':m=quad_candidate_quality(xyz,nodes)
            else:
                q=triangle_quality(xyz,np.asarray([nodes],dtype=np.int64))
                m={name:float(getattr(q,name)[0]) for name in ('scaled_jacobian','aspect_ratio','minimum_angle','maximum_angle','warpage')}
            count+=sum((m['scaled_jacobian']<settings.min_scaled_jacobian,m['aspect_ratio']>settings.max_aspect_ratio,m['minimum_angle']<settings.min_angle,m['maximum_angle']>settings.max_angle,m['warpage']>settings.max_warpage))
            slack.append(min((m['minimum_angle']-settings.min_angle)/(settings.min_angle or 1.),(settings.max_angle-m['maximum_angle'])/((180.-settings.max_angle) or 1.),
                (m['scaled_jacobian']-settings.min_scaled_jacobian)/(abs(settings.min_scaled_jacobian) or 1.),(settings.max_aspect_ratio-m['aspect_ratio'])/settings.max_aspect_ratio,
                (settings.max_warpage-m['warpage'])/(settings.max_warpage or 1.)))
            corners=xyz[list(nodes)];size=float(np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1).mean())
            for edge in _edges(nodes):lengths.setdefault(edge,[]).append(size)
    excess=0.
    for edge,sizes in lengths.items():
        if len(sizes)==1 and edge in outside:sizes=sizes+[outside[edge]]
        if len(sizes)==2 and min(sizes)>0:excess+=max(0.,max(sizes[0]/sizes[1],sizes[1]/sizes[0])-settings.max_element_growth)
    return count,excess,-min(slack)


def _place_new_rows(points,mesh,parents,selected,mids,settings,evaluate,charge,check,records,incidence):
    """Bounded placement of only the rows created by this cut; returns the UV array or None.

    Pre-existing rows stay bit-frozen. Every trial is charged before it is
    evaluated; the caller still applies the unchanged global admission.
    """
    from .errors import GeometryError
    from ._physical_shape_patches import _OriginalStars,_STAR_CALL
    n=len(points);rows=sorted(mids.values())
    if rows!=list(range(n,n+len(selected))) or len(rows)>2 or mesh.node_coordinates.shape[0]!=n:return None
    ids={p[2] for p in parents};outside={}
    for p in parents:
        for edge in _edges(p[3]):
            for i in incidence[edge]:
                if records[i][2] not in ids:outside[edge]=records[i][5]
    base=np.vstack((points,np.asarray([.5*points[a]+.5*points[b] for a,b in selected])))
    def trial(positions):
        charge()
        proposed=np.vstack((points,np.asarray(positions)))
        if any((points==proposed[r]).all(axis=1).any() for r in rows):return None
        check()
        try:fresh=np.asarray(evaluate(proposed[n:]),dtype=float)
        except GeometryError:return None
        check()
        if fresh.shape!=(len(rows),3) or not np.isfinite(fresh).all():return None
        world=np.vstack((mesh.node_coordinates,fresh))
        children=[_children(p,mids,proposed,world,settings,lambda:None,outside) for p in parents]
        if any(c is None for c in children):return None
        if _boundary([p[3] for p in parents])!=_boundary([nodes for c in children for _,nodes in c]):return None
        return _star_objective(children,world,settings,outside),proposed,children
    def sound(proposed,children):
        cells=[(kind,nodes) for c in children for kind,nodes in c]
        triangles=np.asarray([nodes for kind,nodes in cells if kind=='T'],dtype=np.int64).reshape((-1,3))
        quads=np.asarray([nodes for kind,nodes in cells if kind=='Q'],dtype=np.int64).reshape((-1,4))
        star=type('LocalStar',(),dict(triangle_connectivity=triangles,triangle_active=np.ones(len(triangles),dtype=bool),quad_connectivity=quads,quad_active=np.ones(len(quads),dtype=bool)))
        return _STAR_CALL(_OriginalStars(base,star,rows),proposed)
    positions=[base[r].copy() for r in rows];best=None
    def consider(index,position):
        nonlocal best
        trial_positions=list(positions);trial_positions[index]=position
        result=trial(trial_positions)
        if result is None or (best is not None and not _search_order(result[0])<_search_order(best[0])) or not sound(result[1],result[2]):return
        best=result;positions[index]=position
    for index,(a,b) in enumerate(selected):
        for step in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15):
            t=step/16;consider(index,(1.-t)*points[a]+t*points[b])
        if best is not None and best[0][0]==0 and best[0][1]==0.:continue
        if best is None:reach=float(np.linalg.norm(points[b]-points[a]))
        else:
            row=rows[index]
            reach=min(float(np.linalg.norm(best[1][y]-best[1][x])) for c in best[2] for _,nodes in c for x,y in zip(nodes,nodes[1:]+nodes[:1]) if row in (x,y))
        for scale in (.25,.0625,.015625):
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                consider(index,positions[index]+reach*scale*np.asarray([dx,dy],dtype=float))
    if best is None or best[0][0]!=0 or best[1][:n].tobytes()!=points.tobytes():return None
    return best[1]


def _retriangulate_cavity(points,mesh,parents,selected,settings,evaluate,charge,check,records,incidence,protected):
    """Insert one row and re-triangulate a frozen-coordinate T cavity; returns (UV array, cavity cells, child groups) or None.

    The cavity is the two cut parents plus one or two unprotected, same-owner,
    active neighbouring T cells. The new row joins at least four cavity corners;
    interior edges of the old cavity (unprotected) may be replaced, so existing
    edges are flipped but no existing coordinate moves and no boundary changes.
    Every trial is charged; the caller still applies the unchanged global admission.
    """
    from .errors import GeometryError
    from .quality_v2 import triangle_quality
    from ._physical_shape_patches import _OriginalStars,_STAR_CALL
    n=len(points);xyz0=mesh.node_coordinates
    if len(selected)!=1 or len(parents)!=2 or any(p[0]!='T' or p[4]!=parents[0][4] for p in parents) or xyz0.shape[0]!=n or selected[0] in protected:return None
    owner=parents[0][4];ids={p[2] for p in parents};neighbours=[]
    for p in parents:
        for edge in _edges(p[3]):
            if edge==selected[0] or edge in protected or len(incidence[edge])!=2:continue
            for i in incidence[edge]:
                r=records[i]
                if r[2] not in ids and r[0]=='T' and r[4]==owner and _positive(points,r[3]) and all(r[2]!=q[2] for q in neighbours):neighbours.append(r)
    sets=[(r,) for r in neighbours]+[(r,s) for k,r in enumerate(neighbours) for s in neighbours[k+1:]]
    budget=[128];a,b=selected[0]

    def slack(nodes,uv,world):
        if not _positive(uv,nodes):return None
        q=triangle_quality(world,np.asarray([nodes],dtype=np.int64))
        m={name:float(getattr(q,name)[0]) for name in ('scaled_jacobian','aspect_ratio','minimum_angle','maximum_angle','warpage')}
        return min((m['minimum_angle']-settings.min_angle)/(settings.min_angle or 1.),(settings.max_angle-m['maximum_angle'])/((180.-settings.max_angle) or 1.),
            (m['scaled_jacobian']-settings.min_scaled_jacobian)/(abs(settings.min_scaled_jacobian) or 1.),(settings.max_aspect_ratio-m['aspect_ratio'])/settings.max_aspect_ratio,
            (settings.max_warpage-m['warpage'])/(settings.max_warpage or 1.))

    best=None
    for extra in sets:
        cavity=list(parents)+list(extra);cavity_ids={c[2] for c in cavity}
        directed=_boundary([c[3] for c in cavity]);following={x:y for x,y in directed}
        if len(following)!=len(directed) or len(directed)!=len(cavity)+2:continue
        ring=[min(following)]
        while following[ring[-1]]!=ring[0] and len(ring)<=len(directed):ring.append(following[ring[-1]])
        if len(ring)!=len(directed) or len(set(ring))!=len(ring) or len(ring)<5:continue
        count=len(ring);outside={}
        for c in cavity:
            for edge in _edges(c[3]):
                for i in incidence[edge]:
                    if records[i][2] not in cavity_ids:outside[edge]=records[i][5]
        area=sum(float(points[x][0]*points[y][1]-points[y][0]*points[x][1]) for x,y in zip(ring,ring[1:]+ring[:1]))
        inner={e for e,c in Counter(e for x in cavity for e in _edges(x[3])).items() if c==2}
        fills={};strict=[True]
        def fill(path):
            # Best triangulation of the boundary pocket 'path' closed by its end chord; never adds an edge at the new row.
            key=(strict[0],tuple(path))
            if key in fills:return fills[key]
            result=None
            if len(path)==2:result=(float('inf'),[])
            else:
                for k in range(1,len(path)-1):
                    s=slack((path[0],path[k],path[-1]),points,xyz0)
                    if s is None or (strict[0] and s<0):continue
                    left=fill(path[:k+1]);right=fill(path[k:])
                    if left is None or right is None:continue
                    chords=([(path[0],path[k])] if k>1 else [])+([(path[k],path[-1])] if k<len(path)-2 else [])
                    if any(tuple(sorted(c)) in incidence and tuple(sorted(c)) not in inner for c in chords):continue
                    value=(min(s,left[0],right[0]),[(path[0],path[k],path[-1])]+left[1]+right[1])
                    if result is None or value[0]>result[0]:result=value
            fills[key]=result;return result
        def trial(position):
            budget[0]-=1;charge()
            proposed=np.vstack((points,np.asarray([position])))
            if (points==proposed[n]).all(axis=1).any():return None
            check()
            try:fresh=np.asarray(evaluate(proposed[n:]),dtype=float)
            except GeometryError:return None
            check()
            if fresh.shape!=(1,3) or not np.isfinite(fresh).all():return None
            world=np.vstack((xyz0,fresh));cache={}
            for strict[0] in (True,False):
                found=[]
                for mask in range(1<<count):
                    corners=[i for i in range(count) if mask>>i&1]
                    if len(corners)<4:continue
                    value=float('inf');triangles=[]
                    for t,i in enumerate(corners):
                        j=corners[(t+1)%len(corners)];tri=(n,ring[i],ring[j])
                        if tri not in cache:cache[tri]=slack(tri,proposed,world)
                        path=[ring[(i+u)%count] for u in range(((j-i)%count)+1)]
                        pocket=fill(path) if cache[tri] is not None else None
                        if pocket is None or (len(path)>2 and tuple(sorted((path[0],path[-1]))) in incidence and tuple(sorted((path[0],path[-1]))) not in inner):value=None;break
                        value=min(value,cache[tri],pocket[0]);triangles.extend([tri]+pocket[1])
                    if value is None or len(triangles)!=count or (strict[0] and value<0):continue
                    covered=sum(float(proposed[x][0]*proposed[y][1]-proposed[y][0]*proposed[x][1]+proposed[y][0]*proposed[z][1]-proposed[z][0]*proposed[y][1]+proposed[z][0]*proposed[x][1]-proposed[x][0]*proposed[z][1]) for x,y,z in triangles)
                    if abs(covered-area)>1.e-9*abs(area):continue
                    found.append((value,triangles))
                if found:break
            strict[0]=True
            if not found:return None
            # Every shape-clear spoke mask is scored; the lowest (violations, growth excess, -slack) wins, ties by mask order.
            scored=[(_star_objective([[('T',t) for t in triangles]],world,settings,outside),triangles) for value,triangles in found]
            objective,triangles=min(scored,key=lambda item:item[0])
            return objective,proposed,triangles
        def sound(proposed,triangles):
            star=type('LocalStar',(),dict(triangle_connectivity=np.asarray(triangles,dtype=np.int64).reshape((-1,3)),triangle_active=np.ones(len(triangles),dtype=bool),quad_connectivity=np.empty((0,4),dtype=np.int64),quad_active=np.empty(0,dtype=bool)))
            return _STAR_CALL(_OriginalStars(np.vstack((points,np.asarray([.5*points[a]+.5*points[b]]))),star,[n]),proposed)
        local=None
        def consider(position):
            nonlocal local
            if budget[0]<=0:return
            result=trial(position)
            if result is None or (local is not None and not _search_order(result[0])<_search_order(local[0])) or not sound(result[1],result[2]):return
            local=result+(position,)
        for step in (1,2,3):
            t=step/4;consider((1.-t)*points[a]+t*points[b])
        while local is not None and not (local[0][0]==0 and local[0][1]==0.) and budget[0]>0:
            origin=local[3];reach=min(float(np.linalg.norm(points[x]-origin)) for x in ring);before=local[0]
            for scale in (.25,.0625,.015625):
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                    consider(local[3]+reach*scale*np.asarray([dx,dy],dtype=float))
            if local[0]==before:break
        if local is not None and local[0][0]==0 and (best is None or local[0]<best[0][0]):best=(local,cavity)
        if best is not None and best[0][0][1]==0.:break
        if budget[0]<=0:break
    if best is None:return None
    local,cavity=best;triangles=local[2];k=len(cavity)
    return local[1],cavity,[[('T',t)] for t in triangles[:k-1]]+[[('T',t) for t in triangles[k-1:]]]


def _assemble(mesh,covers,parents,children,xyz):
    from .core import MeshCore
    # Copy changed row arrays once; avoid converting every unchanged row to
    # Python objects for each detached trial. New rows remain a small buffer.
    tr=mesh.triangle_connectivity.copy();ta=mesh.triangle_active.copy()
    qa=mesh.quad_active.copy();extra_tr=[];extra_ta=[];extra_ti=[];extra_to=[]
    current=dict(covers);nextid=int(mesh.element_ids.max())+1;newquads=[]
    for parent,group in zip(parents,children):
        kind,row,identifier,_,owner,_=parent
        oldrows=[row] if kind=='T' else list(current.pop(identifier))
        if kind=='Q':qa[row]=False
        pieces=[]
        for childkind,nodes in group:
            if childkind=='T':pieces.append((nodes,True,None))
            else:
                tag=len(newquads);newquads.append([nodes,owner,[]])
                pieces.extend([((nodes[0],nodes[1],nodes[2]),False,tag),((nodes[0],nodes[2],nodes[3]),False,tag)])
        if len(pieces)<len(oldrows):return None
        for j,(nodes,active,tag) in enumerate(pieces):
            if j<len(oldrows):target=oldrows[j];tr[target]=nodes;ta[target]=active
            else:
                target=len(tr)+len(extra_tr);extra_tr.append(nodes);extra_ta.append(active)
                extra_ti.append(nextid);nextid+=1;extra_to.append(owner)
            if tag is not None:newquads[tag][2].append(target)
    qi=[]
    for _,_,rows in newquads:
        qi.append(nextid);current[nextid]=tuple(rows);nextid+=1
    added=len(xyz)-mesh.num_nodes
    def appended(original,values):
        return np.concatenate((original,np.asarray(values,dtype=original.dtype).reshape((-1,)+original.shape[1:])))
    nodeids=appended(mesh.node_ids,np.arange(int(mesh.node_ids.max())+1,int(mesh.node_ids.max())+1+added,dtype=np.int64))
    result=MeshCore(xyz,appended(tr,extra_tr),appended(mesh.quad_connectivity,[nodes for nodes,_,_ in newquads]),node_ids=nodeids,triangle_ids=appended(mesh.triangle_ids,extra_ti),quad_ids=appended(mesh.quad_ids,qi),owner_table=mesh.owner_table,node_owner_handles=appended(mesh.node_owner_handles,np.full(added,-1,dtype=np.int32)),triangle_owner_handles=appended(mesh.triangle_owner_handles,extra_to),quad_owner_handles=appended(mesh.quad_owner_handles,[owner for _,owner,_ in newquads]),node_active=appended(mesh.node_active,np.ones(added,dtype=bool)),triangle_active=appended(ta,extra_ta),quad_active=appended(qa,np.ones(len(newquads),dtype=bool)))
    return result,current


def _score(mesh,settings):
    from .quality_v2 import evaluate_quality
    from .surface_mesh import _quality_threshold_report,_published_quality_report
    q=evaluate_quality(mesh,check_validity=False);threshold=_quality_threshold_report(q,settings)
    report=_published_quality_report(mesh,settings,q,threshold)
    report['violation_counts']=dict(threshold['violation_counts']);report['invalid_element_count']=len(q.validity.errors)
    return report,q


def _covered_report(report,mesh,cover,quality,protected):
    from .errors import MeshError
    active=mesh.quad_ids[mesh.quad_active]
    pairs=np.asarray([[int(mesh.triangle_ids[row]) for row in cover[int(identifier)]] for identifier in active],dtype=np.int64).reshape((-1,2))
    qm=quality.quadrilaterals
    if not np.array_equal(qm.element_ids,active):raise MeshError('mixed sizing quadrilateral IDs differ from quality rows')
    scores=qm.scaled_jacobian-.08*np.log(np.maximum(qm.aspect_ratio,1.))-.002*np.maximum(0.,90.-qm.minimum_angle)-.002*np.maximum(0.,qm.maximum_angle-90.)-.25*qm.warpage
    for array in (pairs,active,scores):array.setflags(write=False)
    result=replace(report,mesh=mesh,paired_triangle_ids=pairs,created_quad_ids=active,scores=scores)
    if _cover(result,protected)!=cover:raise MeshError('mixed sizing produced inconsistent cover metadata')
    return result


def _shape_clear(summary):
    return summary['invalid_element_count']==0 and not any(summary['violation_counts'].values())


def _debt_closure(initial,points,mesh,covers,summary,quality,parents,report,protected,settings,evaluate,charge,check,*,node_remaining,first_work,initial_interfaces,cut_limit=16,node_limit=32,cell_limit=128,work_limit=512,max_fractions=5,new_row_placement=True,local_retriangulation=False):
    """Private connected staging; only the original whole report can admit it.

    A failed transaction returns no geometry or admitted nodes; all work stays spent.
    The first cut was already evaluated by the ordinary single-step path.
    """
    from .errors import MeshError
    from . import surface_mesh as surface
    from ._physical_t3_refinement import _alternative_progress
    if type(max_fractions) is not int or not 1<=max_fractions<=5:raise MeshError('mixed sizing cut fraction count is invalid')
    if type(new_row_placement) is not bool:raise MeshError('mixed sizing row placement switch is invalid')
    if type(local_retriangulation) is not bool:raise MeshError('mixed sizing local retriangulation switch is invalid')
    # Midpoint first; a shape-limited internal cut retries at other fractions of the same edge.
    fractions=(.5,.4,.6,.3,.7)[:max_fractions]
    staged=allocated=node_remaining[0]; allowance=min(node_remaining[1],node_limit)
    touched={p[2] for p in parents}; connected=set(touched);owner=parents[0][4]
    if any(p[4]!=owner for p in parents):raise MeshError('mixed sizing closure crosses owners')
    cuts=1;work=first_work;reason='no_connected_debt'
    exhausted=MeshError('mixed sizing closure work exhausted')
    def spend():
        nonlocal work
        if work>=work_limit:raise exhausted
        charge();work+=1
    facts=dict(cuts=cuts,staged_nodes=staged,allocated_nodes=allocated,admitted_nodes=0,touched_cells=len(touched),work=work,accepted=False,initial_growth=initial['elements_above_maximum_growth'],trajectory=[(summary['elements_above_maximum_growth'],summary['max_element_growth'])])
    try:
        while True:
            check()
            records,incidence,frontier=_frontier(mesh,settings.max_element_growth)
            connected.update(r[2] for r in records if any(n>=node_remaining[2] for n in r[3]))
            touched.update(connected)
            if len(touched)>cell_limit:reason='cell_limit';break
            if _alternative_progress(initial,summary):
                facts.update(accepted=True,stop_reason='original_global_progress',cuts=cuts,staged_nodes=staged,allocated_nodes=allocated,admitted_nodes=staged,touched_cells=len(touched),work=work)
                return (points,mesh,covers,summary,quality),staged,facts
            if cuts>=cut_limit:reason='cut_limit';break
            if staged>=allowance:reason='node_limit';break
            candidates=[];blocked_by_cell=False
            for _,_,interface,large in frontier:
                adjacent=incidence[interface]
                if interface in initial_interfaces and not any(n>=node_remaining[2] for n in records[large][3]):continue
                if not any(records[i][2] in connected for i in adjacent):continue
                parent=records[large];kind,_,identifier,nodes,parent_owner,_=parent
                if parent_owner!=owner:continue
                bad={edge for _,_,edge,index in frontier if index==large}
                def eligible(edge):
                    attached=incidence.get(edge,())
                    return edge not in protected and edge not in bad and len(attached)==2 and all(records[i][4]==owner and _positive(points,records[i][3]) for i in attached)
                edges=_edges(nodes)
                if kind=='T':
                    choices=[(e,) for e in edges if eligible(e)]
                    choices.sort(key=lambda es:(-float(np.linalg.norm(mesh.node_coordinates[es[0][1]]-mesh.node_coordinates[es[0][0]])),es))
                else:
                    choices=[(edges[i],edges[i+2]) for i in range(2) if eligible(edges[i]) and eligible(edges[i+2])]
                    choices.sort(key=lambda es:(-sum(float(np.linalg.norm(mesh.node_coordinates[b]-mesh.node_coordinates[a])) for a,b in es),es))
                for selected in choices:
                    affected=sorted({i for edge in selected for i in incidence[edge]});group=[records[i] for i in affected]
                    if any(sum(e in _edges(p[3]) for e in selected)>1 and p[2]!=identifier for p in group):continue
                    if len(touched|{p[2] for p in group})>cell_limit:blocked_by_cell=True;continue
                    candidates.append((selected,group))
            if not candidates:
                reason='cell_limit' if blocked_by_cell else 'no_connected_debt'
                break
            stepped=None
            for sweep in (0,)+((1,) if new_row_placement else ()):
                for selected,group in candidates:
                    if staged+len(selected)>allowance:reason='node_limit';continue
                    for fraction in (fractions if sweep==0 else (None,)):
                        spend()
                        mids={edge:len(points)+i for i,edge in enumerate(selected)}
                        if fraction is None:
                            if sweep==1:proposed=_place_new_rows(points,mesh,group,selected,mids,settings,evaluate,spend,check,records,incidence)
                            else:
                                found=_retriangulate_cavity(points,mesh,group,selected,settings,evaluate,spend,check,records,incidence,protected)
                                proposed,group,regrouped=found if found is not None else (None,group,None)
                            if proposed is None:continue
                            allocated+=len(selected)
                        else:
                            proposed=np.vstack((points,np.asarray([(1.-fraction)*points[a]+fraction*points[b] for a,b in selected])))
                            allocated+=len(selected)
                            if any((points==proposed[index]).all(axis=1).any() for index in mids.values()):reason='duplicate_midpoint';continue
                        check();world=np.asarray(evaluate(proposed),dtype=float);check()
                        if world[:len(points)].tobytes()!=mesh.node_coordinates.tobytes():raise MeshError('mixed sizing closure owner XYZ prefix changed')
                        children=regrouped if sweep==2 and fraction is None else [_children(p,mids,proposed,world,settings,spend,_neighbour_sizes(group,records,incidence) if new_row_placement else None) for p in group]
                        if any(c is None for c in children):reason='unsupported_template';continue
                        if _boundary([p[3] for p in group])!=_boundary([nodes for child in children for _,nodes in child]):reason='coverage';continue
                        assembled=_assemble(mesh,covers,group,children,world)
                        if assembled is None:reason='cover';continue
                        candidate,cover=assembled
                        if surface._physical_chart_invalid_count(proposed,candidate.triangle_connectivity) or not all(_positive(proposed,tuple(map(int,row))) for row in candidate.quad_connectivity[candidate.quad_active]):reason='source_uv';continue
                        spend();check();result,q=_score(candidate,settings);check()
                        if not _shape_clear(result):reason='intermediate_shape';continue
                        stepped=True;break
                    if stepped:break
                if stepped:break
            if not stepped:break
            cuts+=1;touched.update(p[2] for p in group)
            _covered_report(report,candidate,cover,q,protected)
            staged+=len(selected)
            points,mesh,covers,summary,quality=proposed,candidate,cover,result,q
            facts['trajectory'].append((summary['elements_above_maximum_growth'],summary['max_element_growth']))
    except MeshError as error:
        if error is not exhausted:raise
        reason='transaction_work_limit'
    facts.update(cuts=cuts,staged_nodes=staged,allocated_nodes=allocated,admitted_nodes=0,touched_cells=len(touched),work=work,stop_reason=reason)
    return None,0,facts


def repair_mixed_sizing(points,report,settings,evaluate,protected_edges,*,topology_remaining,insertion_remaining,max_fractions=5,new_row_placement=True,local_retriangulation=False):
    """Return detached UV/report/work/new-node counts; all final gates remain hard."""
    from types import FunctionType
    from . import surface_mesh as surface
    from .errors import MeshError
    from ._physical_t3_refinement import _alternative_progress
    full_builder,full_code=surface._PHYSICAL_FULL_BUILDER_REF,surface._PHYSICAL_FULL_BUILDER_CODE
    pairing_builder,pairing_code=surface._PHYSICAL_PAIRING_BUILDER_REF,surface._PHYSICAL_PAIRING_BUILDER_CODE
    full_services=surface._REFERENCE_PHYSICAL_FULL_SERVICES;pairing_services=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    facts=dict(enabled=False,attempts=0,commits=0,template_work=0,new_nodes=0,attempted_allocation_nodes=0,validated_private_staged_nodes=0,stop_reason='ineligible')
    # A finite direct check precedes every builder call, including this first one.
    if not (surface._physical_full_quality_services is full_builder and type(full_builder) is FunctionType and full_builder.__code__ is full_code and full_builder.__defaults__ is None and full_builder.__kwdefaults__ is None and surface._physical_pairing_services is pairing_builder and type(pairing_builder) is FunctionType and pairing_builder.__code__ is pairing_code and pairing_builder.__defaults__ is None and pairing_builder.__kwdefaults__ is None and full_services is not None and pairing_services is not None
                and full_builder()==full_services and pairing_builder()==pairing_services):return points,report,0,0,facts
    chart=_eligible_chart(evaluate)
    if chart is None:return points,report,0,0,facts
    if type(max_fractions) is not int or not 1<=max_fractions<=5:raise MeshError('mixed sizing cut fraction count is invalid')
    if type(new_row_placement) is not bool:raise MeshError('mixed sizing row placement switch is invalid')
    if type(local_retriangulation) is not bool:raise MeshError('mixed sizing local retriangulation switch is invalid')
    fractions=(.5,.4,.6,.3,.7)[:max_fractions]
    if type(topology_remaining) is not int or topology_remaining<0 or type(insertion_remaining) is not int or insertion_remaining<0:raise MeshError('mixed sizing requires exact remaining work/node allowances')
    binding=chart.region_binding;binding.validate(None);chart.validate_reference()
    original=np.asarray(points,dtype=float);current=original.copy();mesh=report.mesh;protected={tuple(sorted(map(int,e))) for e in protected_edges}
    xyz=np.asarray(evaluate(current),dtype=float)
    if xyz.tobytes()!=mesh.node_coordinates.tobytes():raise MeshError('mixed sizing initial owner coordinates differ')
    covers=_cover(report,protected);before,_=_score(mesh,settings);facts.update(enabled=True,stop_reason=None,initial_quality=before)
    if before['invalid_element_count'] or surface._physical_chart_invalid_count(current,mesh.triangle_connectivity):raise MeshError('mixed sizing requires a valid complete source UV cover')
    used=added=consumed=0;budget_error=MeshError('mixed sizing topology work exhausted')
    def charge():
        nonlocal used
        if used>=topology_remaining:raise budget_error
        used+=1
    def check():
        if not (surface._physical_full_quality_services is full_builder and type(full_builder) is FunctionType and full_builder.__code__ is full_code and full_builder.__defaults__ is None and full_builder.__kwdefaults__ is None and surface._physical_pairing_services is pairing_builder and type(pairing_builder) is FunctionType and pairing_builder.__code__ is pairing_code and pairing_builder.__defaults__ is None and pairing_builder.__kwdefaults__ is None and full_services is not None and pairing_services is not None
                and full_builder()==full_services and pairing_builder()==pairing_services):raise MeshError('mixed sizing calculation changed during recovery')
        if chart.check is not None:raise MeshError('mixed sizing acquired an owner callback')
        chart._current()
    while before['quality_violation_count'] and used<topology_remaining and consumed<insertion_remaining:
        check();records,incidence,frontier=_frontier(mesh,settings.max_element_growth)
        if not frontier:facts['stop_reason']='no_growth_frontier';break
        committed=False
        try:
            for sweep in (0,)+((1,) if new_row_placement else ())+((2,) if local_retriangulation else ()):
                seen=set()
                for _,_,fine_edge,large in frontier:
                    parent=records[large];kind,_,identifier,nodes,owner,_=parent
                    if identifier in seen:continue
                    seen.add(identifier)
                    badinterfaces={edge for _,_,edge,index in frontier if index==large}
                    def eligible(edge):
                        attached=incidence.get(edge,())
                        return (edge not in protected and edge not in badinterfaces and len(attached)==2 and all(records[i][4]==owner and _positive(current,records[i][3]) for i in attached))
                    if kind=='T':
                        edges=[(e,) for e in _edges(nodes) if eligible(e)]
                        choices=sorted(edges,key=lambda es:(-float(np.linalg.norm(mesh.node_coordinates[es[0][1]]-mesh.node_coordinates[es[0][0]])),es))
                    else:
                        edges=_edges(nodes);choices=[(edges[i],edges[i+2]) for i in range(2) if eligible(edges[i]) and eligible(edges[i+2])]
                        choices.sort(key=lambda es:(-sum(float(np.linalg.norm(mesh.node_coordinates[b]-mesh.node_coordinates[a])) for a,b in es),es))
                    if not choices:continue
                    # Geometry-ordered cut families per coarse cell and state; first admissible wins.
                    for selected in choices:
                        if consumed+len(selected)>insertion_remaining:continue
                        for fraction in (fractions if sweep==0 else (None,)):
                            proposal_start=used
                            charge();facts['attempts']+=1
                            affected=sorted({i for edge in selected for i in incidence[edge]})
                            parents=[records[i] for i in affected]
                            if any(sum(edge in _edges(p[3]) for edge in selected)>1 and p[2]!=identifier for p in parents):break
                            mids={edge:len(current)+i for i,edge in enumerate(selected)}
                            if fraction is None:
                                if sweep==1:proposed=_place_new_rows(current,mesh,parents,selected,mids,settings,evaluate,charge,check,records,incidence)
                                else:
                                    found=_retriangulate_cavity(current,mesh,parents,selected,settings,evaluate,charge,check,records,incidence,protected)
                                    proposed,parents,regrouped=found if found is not None else (None,parents,None)
                                if proposed is None:continue
                                facts['attempted_allocation_nodes']+=len(selected)
                            else:
                                proposed=np.vstack((current,np.asarray([(1.-fraction)*current[a]+fraction*current[b] for a,b in selected])))
                                facts['attempted_allocation_nodes']+=len(selected)
                                if any((current==proposed[index]).all(axis=1).any() for index in mids.values()):continue
                            check();world=np.asarray(evaluate(proposed),dtype=float);check()
                            if world[:len(current)].tobytes()!=mesh.node_coordinates.tobytes():raise MeshError('mixed sizing owner XYZ prefix changed')
                            old_used=used;children=regrouped if sweep==2 and fraction is None else [_children(p,mids,proposed,world,settings,charge,_neighbour_sizes(parents,records,incidence) if new_row_placement else None) for p in parents];facts['template_work']+=used-old_used
                            if any(c is None for c in children):continue
                            if _boundary([p[3] for p in parents])!=_boundary([nodes for group in children for _,nodes in group]):continue
                            staged=_assemble(mesh,covers,parents,children,world)
                            if staged is None:continue
                            candidate,cover=staged
                            if surface._physical_chart_invalid_count(proposed,candidate.triangle_connectivity):continue
                            charge();check();summary,q=_score(candidate,settings);check()
                            transaction_nodes=0
                            if not _alternative_progress(before,summary):
                                if not (_shape_clear(before) and _shape_clear(summary)):continue
                                check()
                                outcome,transaction_nodes,detail=_debt_closure(before,proposed,candidate,cover,summary,q,parents,report,protected,settings,evaluate,charge,check,node_remaining=(len(selected),insertion_remaining-consumed,len(current)),first_work=used-proposal_start,initial_interfaces={edge for _,_,edge,_ in frontier},work_limit=min(512,used-proposal_start+topology_remaining-used),max_fractions=max_fractions,new_row_placement=new_row_placement,local_retriangulation=local_retriangulation)
                                consumed+=transaction_nodes
                                facts['attempted_allocation_nodes']+=detail['allocated_nodes']-len(selected)
                                facts['validated_private_staged_nodes']+=detail['staged_nodes']
                                facts.setdefault('closure_transactions',[]).append(detail)
                                if outcome is None:continue
                                proposed,candidate,cover,summary,q=outcome
                            retained=len(proposed)-len(current)
                            report=_covered_report(report,candidate,cover,q,protected)
                            current,mesh,covers,before=proposed,candidate,cover,summary
                            added+=retained
                            if not transaction_nodes:consumed+=retained
                            facts['commits']+=1;committed=True;break
                        if committed:break
                    if committed:break
                if committed:break
            if not committed:facts['stop_reason']='insertion_budget' if consumed>=insertion_remaining else 'no_admissible_mixed_proposal';break
        except MeshError as error:
            if error is not budget_error:raise
            facts['stop_reason']='topology_budget';break
    if before['quality_violation_count']==0:facts['stop_reason']='physical_quality_satisfied'
    elif facts['stop_reason'] is None:facts['stop_reason']='insertion_budget' if consumed>=insertion_remaining else 'topology_budget'
    check();fresh=np.asarray(evaluate(current),dtype=float);check()
    if fresh.tobytes()!=mesh.node_coordinates.tobytes() or current[:len(original)].tobytes()!=original.tobytes():raise MeshError('mixed sizing final owner/input coordinates changed')
    binding.validate(None);chart.validate_reference()
    facts.update(new_nodes=added,admitted_nodes=added,consumed_new_nodes=consumed,topology_work=used,final_quality=before)
    return current,report,used,consumed,facts
