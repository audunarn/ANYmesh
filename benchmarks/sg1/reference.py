"""Structured owner-parameter Q8 references, independent of the quad-first engine."""
from collections import defaultdict
from math import ceil
import numpy as np
from anymesher.mesh import Mesh,Coupling


def structured(fixture,h):
    mesh=Mesh(order='quadratic',geometry_model_id=fixture.geometry.model_id,geometry_revision=fixture.geometry.revision)
    divisions={}
    # Conservative sampled owner-edge arc lengths; equalize opposite edge counts
    # and propagate until all patches have conforming tensor grids.
    for face,p in fixture.patches.items():
        for side,(edge,_) in enumerate(fixture.face_edges[face]):
            t=np.linspace(0,1,65)
            uv=[(x,0) for x in t] if side==0 else [(1,x) for x in t] if side==1 else [(1-x,1) for x in t] if side==2 else [(0,1-x) for x in t]
            xyz=np.array([p.point(*q) for q in uv])
            divisions[edge]=max(divisions.get(edge,1),ceil(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()/h))
    changed=True
    while changed:
        changed=False
        for uses in fixture.face_edges.values():
            for a,b in ((0,2),(1,3)):
                x,y=uses[a][0],uses[b][0]; n=max(divisions[x],divisions[y])
                changed|=divisions[x]!=n or divisions[y]!=n; divisions[x]=divisions[y]=n
    # The member and fixed source probes are at patch centres. Keep those
    # locations on the same kind of tensor-grid vertex at both reference
    # resolutions; odd counts would move a point coupling from a cell interior
    # to a vertex as the grid is refined.
    divisions={edge:n+(n%2) for edge,n in divisions.items()}
    ids={}; next_cell=1; center_nodes={}
    def node(key,point):
        if key not in ids:
            ids[key]=len(ids)+1; mesh.nodes[ids[key]]=np.asarray(point)
        elif np.linalg.norm(mesh.nodes[ids[key]]-point)>1e-9:
            raise ValueError('reference shared source edge mismatch')
        return ids[key]
    for face,p in fixture.patches.items():
        uses=fixture.face_edges[face]; nu=divisions[uses[0][0]]; nv=divisions[uses[1][0]]
        grid={}
        for j in range(2*nv+1):
            for i in range(2*nu+1):
                if i%2 and j%2: continue
                side=None
                if j==0: side,index,total=0,i,2*nu
                elif i==2*nu: side,index,total=1,j,2*nv
                elif j==2*nv: side,index,total=2,2*nu-i,2*nu
                elif i==0: side,index,total=3,2*nv-j,2*nv
                if side is not None:
                    edge,forward=uses[side]; canonical=index if forward else total-index
                    entity=fixture.geometry.edges[edge]
                    key=('vertex',entity.start if canonical==0 else entity.end) if canonical in (0,total) else ('edge',edge,canonical,total)
                else: key=('face',face,i,j)
                n=node(key,p.point(i/(2*nu),j/(2*nv))); grid[i,j]=n
                if key[0]=='vertex': mesh.node_of_vertex[key[1]]=n
        center_nodes[face]=grid[nu,nv]
        mesh.elements_of_face[face]=[]
        for j in range(nv):
            for i in range(nu):
                x,y=2*i,2*j
                b=tuple(grid[a,b] for a,b in ((x,y),(x+2,y),(x+2,y+2),(x,y+2),(x+1,y),(x+2,y+1),(x+1,y+2),(x,y+1)))
                mesh.quads[next_cell]=b; mesh.elements_of_face[face].append(next_cell); next_cell+=1
        chains=([grid[i,0] for i in range(2*nu+1)],[grid[2*nu,j] for j in range(2*nv+1)],
                [grid[i,2*nv] for i in range(2*nu,-1,-1)],[grid[0,j] for j in range(2*nv,-1,-1)])
        for (edge,forward),chain in zip(uses,chains):
            chain=chain if forward else chain[::-1]
            if edge in mesh.nodes_of_edge and mesh.nodes_of_edge[edge]!=chain: raise ValueError('reference station mismatch')
            mesh.nodes_of_edge[edge]=chain
    mesh.elements_of_sheet[fixture.sheet]=list(mesh.quads)
    if fixture.members:
        mid=fixture.members[0]; edge=fixture.geometry.edges[fixture.member_edge]
        a=fixture.geometry.vertex_position(edge.start); b=fixture.geometry.vertex_position(edge.end)
        divisions=2*ceil(np.linalg.norm(b-a)/h/2)
        chain=[]
        for i,t in enumerate(np.linspace(0,1,2*divisions+1)):
            key=('vertex',edge.start if i==0 else edge.end) if i in (0,2*divisions) else ('member',mid,i)
            n=node(key,(1-t)*a+t*b); chain.append(n)
            if key[0]=='vertex': mesh.node_of_vertex[key[1]]=n
        mesh.nodes_of_edge[fixture.member_edge]=chain; mesh.nodes_of_member[mid]=chain; mesh.elements_of_member[mid]=[]
        for i in range(divisions):
            mesh.beams[next_cell]=tuple(chain[2*i:2*i+3]); mesh.elements_of_member[mid].append(next_cell); next_cell+=1
        slave=chain[divisions]; face=fixture.faces[0]
        target=(a+b)/2
        center_node=center_nodes[face]
        if np.linalg.norm(mesh.nodes[center_node]-target)>1e-9:
            raise ValueError('reference attachment misses source centre')
        for eid in mesh.elements_of_face[face]:
            body=mesh.quads[eid]
            if center_node not in body: continue
            weights=tuple(1.0 if n==center_node else 0.0 for n in body)
            mesh.couplings[next_cell]=Coupling(slave,body,weights,tuple(target-mesh.nodes[center_node])); break
        if not mesh.couplings: raise ValueError('reference attachment not located')
    return mesh
