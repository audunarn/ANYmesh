"""Physical cylinder material must not inherit specialized hole limits."""
import math
import numpy as np
import pytest
from anygeometry import GeometryModel, Cylinder, OrientedEdge, trim_face, to_dict
from anymesher.hybrid import generate_hybrid_mesh_result

@pytest.mark.parametrize('count',(1,2,4))
def test_multiple_legacy_arc_holes_remain_on_exact_material(count):
    model=GeometryModel()
    surface=Cylinder((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),1.,3.,0.,math.pi/2)
    def loop(points):
        vertices=model.add_points(tuple(surface.evaluate(*point) for point in points))
        edges=[]
        for i,(a,b) in enumerate(zip(points,points[1:]+points[:1])):
            if a[1]==b[1]:
                middle=model.add_point(*surface.evaluate((a[0]+b[0])/2,a[1]))
                edge=model.add_arc(vertices[i],middle,vertices[(i+1)%4])
            else:edge=model.add_line(vertices[i],vertices[(i+1)%4])
            edges.append(OrientedEdge(edge,True))
        return tuple(edges)
    face=model.add_face_from_loop(loop(((0.,0.),(1.,0.),(1.,1.),(0.,1.))),surface=surface)
    starts=tuple((i+.3)/count for i in range(count))
    trim_face(model,face,tuple(loop(((.3,v),(.3,v+.4/count),(.7,v+.4/count),(.7,v))) for v in starts))
    model.add_sheet((face,))
    before=to_dict(model)
    mesh=generate_hybrid_mesh_result(model,target_size=.5,strategy='native',native_backend='python',recombine=False).mesh
    assert to_dict(model)==before
    area=0.
    for element in mesh.elements_of_face[face]:
        xyz=np.asarray([mesh.nodes[node] for node in mesh.corners_of(element)])
        np.testing.assert_allclose(np.linalg.norm(xyz[:,:2],axis=1),1.,rtol=0.,atol=1e-10)
        angle=np.unwrap(np.arctan2(xyz[:,1],xyz[:,0]))
        area+=abs(np.dot(angle,np.roll(xyz[:,2],-1))-np.dot(xyz[:,2],np.roll(angle,-1)))/2
        center=xyz.mean(axis=0);u=math.atan2(center[1],center[0])/(math.pi/2);v=center[2]/3
        assert not any(.3+1e-8<u<.7-1e-8 and start+1e-8<v<start+.4/count-1e-8 for start in starts)
    assert abs(area-1.5*math.pi*.84)<1e-10
