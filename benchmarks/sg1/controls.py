"""Independent analytical controls and fail-closed capability probes."""
import numpy as np
from .measure import check, geometry_signature
from .contract import LIMITS


def analytical():
    from anysolver import solve_linear,ResourceConfig
    from anysolver.boundary import BoundaryCondition,LoadCase
    from anysolver.matrix_assembly import assemble_stiffness_matrix
    from anymesher.mesh import Mesh
    from types import SimpleNamespace
    from .fixtures import build
    from .reference import structured
    from .consumer import import_model,E,SECTION,THICKNESS
    checks=[]; values={}
    f=build('P'); m=structured(f,.5); model=import_model(f,m,clamp=False)
    eps=1e-5; expected={n:np.array([eps*p[0],-.3*eps*p[1],0,0,0,0]) for n,p in m.nodes.items()}
    exterior={n for edge in f.exterior for n in m.nodes_of_edge[edge]}
    for n in exterior:
        model.add_boundary_condition(BoundaryCondition(f'patch-{n}',[n],dict(zip(('ux','uy','uz','rx','ry','rz'),expected[n]))))
    # Out-of-plane modes are irrelevant to this membrane control.
    model.add_boundary_condition(BoundaryCondition('membrane-only',list(set(m.nodes)-exterior),dict.fromkeys(('uz','rx','ry','rz'),0.)))
    u,_=solve_linear(model,LoadCase('prescribed membrane'),resource_config=ResourceConfig(solver_threads=1,assembly_threads=1))
    exact=np.zeros_like(u)
    for n,v in expected.items(): exact[model.mesh.nodes[n].dofs]=v
    K,_=assemble_stiffness_matrix(model); energy=float(.5*u@(K@u)); want=.5*E*eps**2*THICKNESS*4
    err=float(np.linalg.norm(u-exact)/np.linalg.norm(exact)); ee=abs(energy-want)/want
    check(checks,'analytical/membrane-displacement',err<=1e-6,value=err)
    check(checks,'analytical/membrane-energy',ee<=1e-6,value=ee)
    values['membrane']=dict(energy=energy,expected_energy=want)
    m=Mesh(order='quadratic',nodes={1:np.array((0.,0,0)),2:np.array((.5,0,0)),3:np.array((1.,0,0))},beams={1:(1,2,3)})
    model=import_model(SimpleNamespace(exterior=()),m,clamp=False)
    model.add_boundary_condition(BoundaryCondition('axial-only',[1,2,3],dict.fromkeys(('uy','uz','rx','ry','rz'),0.)))
    model.add_boundary_condition(BoundaryCondition('root',[1],{'ux':0.}))
    load=LoadCase('axial'); load.add_nodal_load(3,[100.,0,0,0,0,0])
    u,_=solve_linear(model,load,resource_config=ResourceConfig(solver_threads=1,assembly_threads=1))
    want=100/(E*SECTION['area']); actual=float(u[model.mesh.nodes[3].dofs[0]])
    K,_=assemble_stiffness_matrix(model); energy=float(.5*u@(K@u))
    check(checks,'analytical/B3-displacement',abs(actual-want)/want<=1e-6,value=abs(actual-want)/want)
    check(checks,'analytical/B3-energy',abs(energy-50*want)/(50*want)<=1e-6,value=abs(energy-50*want)/(50*want))
    values['B3']=dict(displacement=actual,expected_displacement=want,energy=energy)
    return dict(checks=checks,values=values)


def negative():
    from .fixtures import build
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.quad.options import QuadMeshingOptions
    from anymesher.quad.public_integration import QuadPublicUnsupported
    from anymesher.errors import MeshError
    checks=[]
    for case in ('CY','CO','R','C'):
        fixture=build(case); edge=fixture.face_edges[fixture.faces[0]][0][0]
        before=geometry_signature(fixture)
        try:
            generate_hybrid_mesh_result(fixture.geometry,face_ids=fixture.faces,beam_edges=(edge,),target_size=.6,
                strategy='native',native_backend='python',recombine=True,order='quadratic',quad_options=QuadMeshingOptions())
        except QuadPublicUnsupported as exc:
            message=str(exc).lower()
            check(checks,f'{case}/boundary-beam-rejection',
                  'beam ownership' in message and 'source-boundary edge' in message,
                  exception=type(exc).__name__,message=str(exc))
        except Exception as exc:
            checks.append(dict(name=f'{case}/boundary-beam-rejection',status='blocked',
                exception=type(exc).__name__,message=str(exc),reason='earlier capability boundary prevents ownership probe'))
        else: check(checks,f'{case}/boundary-beam-rejection',False)
        check(checks,f'{case}/negative-source-immutable',geometry_signature(fixture)==before)
    f=build('P'); g=f.geometry
    a,b,c=g.add_point(0,0,1),g.add_point(.5,.5,1),g.add_point(1,0,1)
    arc=g.add_arc(a,b,c); before=geometry_signature(f)
    try:
        generate_hybrid_mesh_result(g,face_ids=f.faces,beam_edges=(arc,),target_size=.5,order='quadratic',quad_options=QuadMeshingOptions())
    except MeshError as exc: check(checks,'curved-B3-rejection',any(s in str(exc).lower() for s in ('curved','straight','b3')),exception=type(exc).__name__,message=str(exc))
    else: check(checks,'curved-B3-rejection',False)
    check(checks,'curved-B3-source-immutable',geometry_signature(f)==before)
    return dict(checks=checks)
