"""Atomic interior refinement in a certified material chart.

Boundary stations and geometry topology are immutable. New interior points
are evaluated by the geometry owner, and ordinary qualified-S3 admission still
decides whether the complete mesh can be used.
"""
from collections import Counter
import numpy as np

from anygeometry import GeometryError,query_trimmed_surface_charts,evaluate_trimmed_surface_chart
from .errors import MeshError
from .optimization import local_edge_flip
from .quality_v2 import triangle_quality
from .s3_quality import DEFAULT_S3_QUALITY_POLICY,QUALITY_COMPARISON_TOLERANCE


def _edges(nodes):
    return tuple(tuple(sorted((nodes[i],nodes[(i+1)%3]))) for i in range(3))


def _signed_areas(points,rows):
    corners=points[rows]
    first,second=corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]
    return first[:,0]*second[:,1]-first[:,1]*second[:,0]


def _shape_failures(coordinates,rows):
    quality=triangle_quality(coordinates,rows)
    corners=coordinates[rows]
    lengths=np.linalg.norm(corners[:,(1,2,0)]-corners,axis=2)
    denominator=(lengths*lengths).sum(axis=1)
    normalized=np.divide(4*np.sqrt(3.)*quality.area,denominator,
                         out=np.zeros_like(quality.area),where=denominator>0)
    policy=DEFAULT_S3_QUALITY_POLICY
    tol=QUALITY_COMPARISON_TOLERANCE
    failures=((quality.minimum_angle<policy.minimum_angle_deg-tol)
              | (quality.maximum_angle>policy.maximum_angle_deg+tol)
              | (quality.aspect_ratio>policy.maximum_edge_ratio+tol)
              | (quality.scaled_jacobian<policy.minimum_scaled_jacobian-tol)
              | (normalized<policy.minimum_normalized_area-tol))
    if not np.all(np.isfinite(lengths)) or np.any(quality.area<=0):
        raise MeshError('owner-chart refinement requires valid finite triangles')
    return failures


def _shape_score(coordinates,rows):
    """Use the admission floors as an objective, without relaxing any floor."""
    failures=_shape_failures(coordinates,rows)
    quality=triangle_quality(coordinates,rows)
    corners=coordinates[rows]
    squared=np.sum((corners[:,(1,2,0)]-corners)**2,axis=2).sum(axis=1)
    normalized=4*np.sqrt(3.)*quality.area/squared
    policy=DEFAULT_S3_QUALITY_POLICY
    deficits=np.stack((
        1-quality.minimum_angle/policy.minimum_angle_deg,
        quality.maximum_angle/policy.maximum_angle_deg-1,
        quality.aspect_ratio/policy.maximum_edge_ratio-1,
        1-quality.scaled_jacobian/policy.minimum_scaled_jacobian,
        1-normalized/policy.minimum_normalized_area))
    return int(failures.sum()),float(np.sum(np.maximum(deficits,0.)**2))


def refine_triangle_face(mesh,geometry,face_id,chart,*,protected_edges,cache,
                         max_insertions,max_work,cancellation_check=None):
    """Stage interior relocation, conforming bisections and flips atomically."""
    elements={element:tuple(mesh.tris[element]) for element in mesh.elements_of_face[face_id]}
    ids=sorted({node for nodes in elements.values() for node in nodes})
    positions={node:np.asarray(mesh.nodes[node],dtype=float).copy() for node in ids}
    scale=np.asarray((chart.circumferential_length,chart.axial_length))
    coordinates=dict(zip(ids,geometry.face_local_uv_many(face_id,np.asarray([positions[n] for n in ids]))*scale))
    original_local={node:index for index,node in enumerate(ids)}
    original_area=float(_signed_areas(np.asarray([coordinates[n] for n in ids]),np.asarray(
        [[original_local[node] for node in nodes] for nodes in elements.values()])).sum())
    protected={tuple(sorted(edge)) for edge in protected_edges}
    before=Counter(edge for nodes in elements.values() for edge in _edges(nodes))
    boundary={edge for edge,count in before.items() if count==1}
    protected.update(boundary)
    fixed={node for edge in protected for node in edge}
    fixed.update(node for sequence in mesh.nodes_of_edge.values() for node in sequence)
    fixed.update(mesh.node_of_vertex.values())
    fixed.update(node for nodes in mesh.beams.values() for node in nodes)
    for other_face,other_elements in mesh.elements_of_face.items():
        if other_face!=face_id:
            fixed.update(node for element in other_elements
                         for node in (*mesh.tris.get(element,()),*mesh.quads.get(element,())))
    owner=getattr(geometry,'source',geometry)
    evidence=None
    next_node=max(mesh.nodes,default=0)+1
    next_element=max((*mesh.tris,*mesh.quads,*mesh.beams,*mesh.couplings),default=0)+1
    added_nodes={}
    moved_nodes={}
    parents={}
    work=0
    while True:
        if cancellation_check is not None:cancellation_check(f'cylinder chart refinement {face_id}')
        ids=sorted(coordinates); local={node:index for index,node in enumerate(ids)}
        element_ids=sorted(elements)
        rows=np.asarray([[local[node] for node in elements[element]] for element in element_ids],dtype=np.int64)
        physical=np.asarray([positions[node] for node in ids])
        failures=_shape_failures(physical,rows)
        if not failures.any():break
        if len(added_nodes)>=max_insertions:
            raise MeshError('cylinder chart refinement insertion budget exhausted')
        work+=len(elements)
        if work>max_work:
            raise MeshError('cylinder chart refinement work budget exhausted')
        # Laplacian smoothing can pull a vertex toward a short protected
        # boundary. Improve the physical admission objective locally instead.
        # Only this face's unconstrained nodes may move, and every trial stays
        # inside its oriented triangle star and certified owner material.
        relocated=False
        points=np.asarray([coordinates[node] for node in ids])
        signs=_signed_areas(points,rows)
        movable=sorted(set(rows[failures].ravel())-{local[node] for node in fixed if node in local})
        directions=np.asarray(((1,0),(-1,0),(0,1),(0,-1),
                               (1,1),(1,-1),(-1,1),(-1,-1)),dtype=float)
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        for index in movable:
            if cancellation_check is not None:cancellation_check(f'cylinder chart relocation {face_id}')
            incident=np.flatnonzero(np.any(rows==index,axis=1))
            neighbours=np.setdiff1d(np.unique(rows[incident]),(index,))
            radius=.2*float(np.median(np.linalg.norm(points[neighbours]-points[index],axis=1)))
            best_score=_shape_score(physical,rows[incident]);best=None
            for fraction in (1.,.5,.25):
                for direction in directions:
                    work+=len(incident)
                    if work>max_work:raise MeshError('cylinder chart refinement work budget exhausted')
                    trial=points.copy();trial[index]+=radius*fraction*direction
                    if not np.all(_signed_areas(trial,rows[incident])*signs[incident]>0):continue
                    if evidence is None:evidence=query_trimmed_surface_charts(owner,(face_id,))
                    try:
                        xyz=evaluate_trimmed_surface_chart(owner,evidence,face_id,
                            (trial[index]/scale).reshape(1,2),require_material=True,
                            cancellation_check=cancellation_check)[0]
                    except GeometryError as error:
                        if str(error)!='trimmed chart sample is outside material':raise
                        continue
                    lifted=physical.copy();lifted[index]=xyz
                    score=_shape_score(lifted,rows[incident])
                    if score<best_score:
                        best_score=score;best=(trial[index].copy(),xyz.copy())
            if best is not None:
                node=ids[index];coordinates[node],positions[node]=best
                points[index],physical[index]=best
                if node in added_nodes:added_nodes[node]=best[1].copy()
                else:moved_nodes[node]=best[1].copy()
                relocated=True
        if relocated:continue
        incidence={}
        for element,nodes in elements.items():
            for edge in _edges(nodes):incidence.setdefault(edge,[]).append(element)
        candidates={edge for element in np.asarray(element_ids)[failures] for edge in _edges(elements[int(element)])}
        candidates={edge for edge in candidates-protected if len(incidence[edge])==2}
        if not candidates:
            raise MeshError('cylinder chart has no eligible interior refinement edge')
        edge=min(candidates,key=lambda edge:(-np.linalg.norm(coordinates[edge[0]]-coordinates[edge[1]]),edge))
        point=.5*(coordinates[edge[0]]+coordinates[edge[1]])
        if np.array_equal(point,coordinates[edge[0]]) or np.array_equal(point,coordinates[edge[1]]):
            raise MeshError('cylinder chart refinement cannot resolve an interior midpoint')
        if evidence is None:evidence=query_trimmed_surface_charts(owner,(face_id,))
        xyz=evaluate_trimmed_surface_chart(owner,evidence,face_id,(point/scale).reshape(1,2),
            require_material=True,cancellation_check=cancellation_check)[0]
        coordinates[next_node]=point;positions[next_node]=xyz;added_nodes[next_node]=xyz.copy()
        for element in incidence[edge]:
            nodes=elements[element]
            index=next(i for i in range(3) if tuple(sorted((nodes[i],nodes[(i+1)%3])))==edge)
            a,b,c=(nodes[(index+j)%3] for j in range(3))
            elements[element]=(a,next_node,c)
            elements[next_element]=(next_node,b,c)
            parents[next_element]=parents.get(element,element)
            next_element+=1
        next_node+=1
        ids=sorted(coordinates);local={node:index for index,node in enumerate(ids)}
        element_ids=sorted(elements)
        points=np.asarray([coordinates[node] for node in ids])
        rows=np.asarray([[local[node] for node in elements[element]] for element in element_ids],dtype=np.int64)
        signs=_signed_areas(points,rows)
        if not (np.all(signs>0) or np.all(signs<0)):
            raise MeshError('cylinder chart refinement has inconsistent orientation')
        remaining=max_work-work
        if remaining<1:raise MeshError('cylinder chart refinement work budget exhausted')
        result=local_edge_flip(points,rows,
            protected_edges=[(local[a],local[b]) for a,b in protected if a in local and b in local],
            max_flips=remaining)
        work+=int(result.queue_visits)
        if work>max_work:raise MeshError('cylinder chart refinement work budget exhausted')
        made=np.asarray(result.triangles)
        if (made.shape!=rows.shape or not np.issubdtype(made.dtype,np.integer)
                or np.any(made<0) or np.any(made>=len(ids))
                or np.any(np.diff(np.sort(made,axis=1),axis=1)==0)):
            raise MeshError('cylinder chart refinement returned invalid topology')
        if np.all(signs<0):made=made[:,(0,2,1)]
        updated_signs=_signed_areas(points,made)
        if not np.all(updated_signs*signs[0]>0):
            raise MeshError('cylinder chart refinement reversed material orientation')
        elements={element:tuple(ids[index] for index in row) for element,row in zip(element_ids,made)}
    after=Counter(edge for nodes in elements.values() for edge in _edges(nodes))
    if any(count>2 for count in after.values()) or len({tuple(sorted(nodes)) for nodes in elements.values()})!=len(elements):
        raise MeshError('cylinder chart refinement returned nonmanifold topology')
    if {edge for edge,count in after.items() if count==1}!=boundary or any(
            after.get(edge)!=before[edge] for edge in protected if edge in before):
        raise MeshError('cylinder chart refinement changed protected topology')
    final_area=float(_signed_areas(np.asarray([coordinates[n] for n in ids]),np.asarray(
        [[local[node] for node in nodes] for nodes in elements.values()])).sum())
    if not np.isclose(final_area,original_area,rtol=1e-12,atol=0.):
        raise MeshError('cylinder chart refinement changed material coverage')
    if cancellation_check is not None:cancellation_check('cylinder chart refinement before publish')
    if added_nodes or moved_nodes:
        mesh.nodes.update(moved_nodes)
        mesh.nodes.update(added_nodes)
        mesh.tris.update(elements)
        mesh.elements_of_face[face_id]=sorted(elements)
        for mapping in (mesh.elements_of_sheet,mesh.elements_of_member,mesh.elements_of_edge):
            for values in mapping.values():
                original=set(values)
                values.extend(element for element,parent in parents.items() if parent in original)
        for element,parent in parents.items():
            if parent in mesh.activity:mesh.activity[element]=mesh.activity[parent]
        index={}
        for element,nodes in elements.items():
            for edge in _edges(nodes):index.setdefault(edge,set()).add(element)
        cache[face_id]=index
    return {'insertions':len(added_nodes),'relocated_nodes':len(moved_nodes),
            'added_elements':len(parents),'work':work,
            'protected_coordinates_changed':False}
