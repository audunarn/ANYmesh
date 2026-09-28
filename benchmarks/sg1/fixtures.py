"""Identity-built SG1 assemblies and independently parameterized owner patches."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from anygeometry import GeometryModel, Cylinder, Cone, RuledSurface, CoonsSurface
from anygeometry.entities import OrientedEdge
from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange
from anymesher.refinement import Refinement, SizeField

CASES = ('P', 'F', 'CY', 'CO', 'R', 'C', 'T-R', 'T-C')


@dataclass
class Patch:
    surface: object
    # Geometry surfaces are parameterized independently on each physical patch.
    def point(self, u, v):
        return np.asarray(self.surface.evaluate(float(u), float(v)), dtype=float)

    def uv(self, point):
        return np.asarray(self.surface.local_uv(tuple(point)), dtype=float)

    def derivatives(self, u, v):
        # Central differentiation of the analytic owner, independent of mesh mapping.
        eps = 1.e-5
        du = (self.point(u+eps, v)-self.point(u-eps, v))/(2*eps)
        dv = (self.point(u, v+eps)-self.point(u, v-eps))/(2*eps)
        return du, dv

    def normal(self, u, v):
        du, dv = self.derivatives(u, v)
        n = np.cross(du, dv)
        return n / np.linalg.norm(n)


@dataclass
class Fixture:
    case: str
    geometry: GeometryModel
    patches: dict[int, Patch]
    face_edges: dict[int, tuple[tuple[int, bool], ...]]
    shared: tuple[int, ...]
    exterior: tuple[int, ...]
    center: np.ndarray
    members: tuple[int, ...] = ()
    member_edge: int | None = None
    member_axis: np.ndarray | None = None
    member_end: int | None = None
    sheet: int | None = None

    @property
    def faces(self):
        return tuple(self.patches)

    @property
    def sizes(self):
        return (.5, .25, .125) if self.case in ('P', 'F') else (.6, .3, .15)

    def refinements(self, h, graded):
        if not graded and not self.case.startswith('T-'):
            return ()
        return (Refinement(size=h/3, radius=.35, center=tuple(self.center), growth=1.5, name='SG1'),)

    def field(self, h, graded):
        return SizeField(self.geometry, h, self.refinements(h, graded))


def _bilinear(points, coons=False):
    a,b,c,d = np.asarray(points, dtype=float)
    if coons:
        return CoonsSurface(bottom=np.array((a,b)), right=np.array((b,c)),
                            top=np.array((d,c)), left=np.array((a,d)))
    return RuledSurface(np.array((a,b)), np.array((d,c)))


def build(case: str, member=False, *, member_uv=(.5, .5)) -> Fixture:
    if case not in CASES:
        raise ValueError(case)
    g = GeometryModel()
    vertices = {}  # Explicit symbolic topology keys, never coordinate keys.
    edges = {}
    patches, uses = {}, {}

    def add_patch(keys, surface, arcs=()):
        patch = Patch(surface)
        uv = ((0,0),(1,0),(1,1),(0,1))
        for key, (u,v) in zip(keys, uv):
            if key not in vertices:
                vertices[key] = g.add_point(*patch.point(u,v))
        loop = []
        for i in range(4):
            a,b = keys[i], keys[(i+1)%4]
            if (b,a) in edges:
                eid, forward = edges[b,a], False
            elif (a,b) in edges:
                eid, forward = edges[a,b], True
            else:
                if i in arcs:
                    mid = np.mean([uv[i],uv[(i+1)%4]],axis=0)
                    vm = g.add_point(*patch.point(*mid))
                    eid = g.add_arc(vertices[a], vm, vertices[b])
                else:
                    eid = g.add_line(vertices[a], vertices[b])
                edges[a,b] = eid
                forward = True
            loop.append((eid,forward))
        fid = g.add_face_from_loop(tuple(OrientedEdge(e,f) for e,f in loop),
                                   corners=(0,1,2,3), surface=surface)
        patches[fid], uses[fid] = patch, tuple(loop)
        return fid

    if case in ('P','F'):
        add_patch(('a','b','c','d'), _bilinear(((0,0,0),(2,0,0),(2,1,0),(0,1,0))))
        end = ((4,0,0),(4,1,0)) if case == 'P' else ((2,0,2),(2,1,2))
        add_patch(('b','e','f','c'), _bilinear(((2,0,0),end[0],end[1],(2,1,0))))
    elif case in ('CY','CO'):
        for i in range(2):
            args = (np.zeros(3),np.array((0.,0.,1.)),np.array((1.,0.,0.)))
            s = (Cylinder(*args,2.,3.,i*np.pi/4,np.pi/4) if case == 'CY'
                 else Cone(*args,2.,1.,3.,i*np.pi/4,np.pi/4))
            add_patch((f'b{i}',f'b{i+1}',f't{i+1}',f't{i}'),s,arcs=(0,2))
        if case == 'CY':
            # Plate extends along the tangent away from the angular sector.
            add_patch(('b0','t0','pt','pb'),_bilinear(((2,0,0),(2,0,3),(2,-1,3),(2,-1,0))))
    else:
        coons = case in ('C','T-C')
        if case.startswith('T-'):
            add_patch(('a','b','c','d'),_bilinear(((0,0,0),(2,0,0),(2,1,.5),(0,1,0)),coons))
        else:
            for i in range(2):
                x0,x1 = float(i),float(i+1)
                add_patch((f'b{i}',f'b{i+1}',f't{i+1}',f't{i}'),
                          _bilinear(((x0,0,0),(x1,0,0),(x1,1,.25*x1),(x0,1,.25*x0)),coons))
    counts = Counter(e for loop in uses.values() for e,_ in loop)
    shared = tuple(e for e,c in counts.items() if c==2)
    exterior = tuple(e for e,c in counts.items() if c==1)
    first = next(iter(patches.values()))
    center = first.point(1,.5) if shared else first.point(.5,.5)
    part = g.add_part(name='SG1-'+case)
    sheet = g.add_sheet(tuple(patches), part_id=part)
    fixture = Fixture(case,g,patches,uses,shared,exterior,center,sheet=sheet)
    if member:
        if case in ('P','F') or case.startswith('T-'):
            raise ValueError('members belong to CY/CO/R/C')
        face = next(iter(patches))
        p = patches[face]
        if not (0.0 < member_uv[0] < 1.0 and 0.0 < member_uv[1] < 1.0):
            raise ValueError('member_uv must be interior to the owner face')
        xyz, normal = p.point(*member_uv), p.normal(*member_uv)
        a,b = g.add_point(*(xyz-.5*normal)),g.add_point(*(xyz+.5*normal))
        edge = g.add_line(a,b)
        mid = g.add_member((edge,),part_id=part)
        g.add_attachment(mid, AttachmentKind.MEMBER_THROUGH_FACE, AttachmentTargetKind.FACE,
                         face, ParameterRange.point(.5),
                         (ParameterRange.point(member_uv[0]),ParameterRange.point(member_uv[1])))
        fixture.members,fixture.member_edge,fixture.member_axis,fixture.member_end = (mid,),edge,normal,b
    return fixture


def generate(fixture, h, graded=False, order='quadratic', cancellation_check=None):
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.quad.options import QuadMeshingOptions
    return generate_hybrid_mesh_result(fixture.geometry, face_ids=fixture.faces,
        member_ids=fixture.members, target_size=h, strategy='native', native_backend='python',
        recombine=True, order=order,
        quad_options=QuadMeshingOptions(quality_model='shape_jacobian'),
        refinements=fixture.refinements(h,graded), cancellation_check=cancellation_check).mesh
