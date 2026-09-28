"""Gate-local, explicit legacy Q8/T6 consumer. No solver implementation changes."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
from anymesher.quad.high_order import shape_values, evaluate_mapping
from .measure import quadrature, check
from .contract import LIMITS

E=210e9
THICKNESS=.02
SECTION=dict(area=.05**2,Iy=.05**4/12,Iz=.05**4/12,J=.1406*.05**4)


def import_model(fixture,mesh,clamp=True):
    from anysolver import FEModel
    from anysolver.elements import create_shell_element,QuadraticBeamElement
    from anysolver.mesh_gen import InterpolatedBeamShellMPCElement
    from anysolver.boundary import BoundaryCondition
    model=FEModel('SG1'); model.add_material('SG1',E,.3)
    for n,p in mesh.nodes.items(): model.add_node(n,*p)
    for e,b in {**mesh.quads,**mesh.tris}.items():
        if len(b) not in (8,6): raise ValueError('SG1 solver only consumes Q8/T6')
        model.add_element(e,create_shell_element(e,list(b),material_name='SG1',thickness=THICKNESS,formulation='legacy'))
    for e,b in mesh.beams.items():
        if len(b)!=3: raise ValueError('SG1 solver only consumes B3')
        model.add_element(e,QuadraticBeamElement(e,list(b),material_name='SG1',cross_section=SECTION.copy()))
    for e,c in mesh.couplings.items():
        model.add_element(e,InterpolatedBeamShellMPCElement(e,c.beam_node,list(c.plate_nodes),
            np.asarray(c.weights),np.asarray(c.eccentricity),material_name='SG1'))
    if clamp:
        nodes=sorted({n for edge in fixture.exterior for n in mesh.nodes_of_edge[edge]})
        model.add_boundary_condition(BoundaryCondition('SG1 exterior clamp',nodes,dict.fromkeys(('ux','uy','uz','rx','ry','rz'),0.)))
    # Verify the actual constructed records before allowing assembly.
    for n,p in mesh.nodes.items():
        if not np.array_equal(model.mesh.nodes[n].coords(),p): raise ValueError('consumer changed node')
    for e,b in {**mesh.quads,**mesh.tris,**mesh.beams}.items():
        element=model.mesh.elements[e]
        if tuple(element.node_ids)!=tuple(b) or element.material_name!='SG1': raise ValueError('consumer changed element')
        if e not in mesh.beams and element.thickness!=THICKNESS: raise ValueError('consumer changed thickness')
        if e in mesh.beams and element.cross_section!=SECTION: raise ValueError('consumer changed section')
    for e,c in mesh.couplings.items():
        element=model.mesh.elements[e]
        if (element.beam_node_id!=c.beam_node or tuple(element.shell_node_ids)!=c.plate_nodes
            or not np.array_equal(element.shape_weights,c.weights) or not np.array_equal(element.eccentricity,c.eccentricity)):
            raise ValueError('consumer changed coupling')
    if len(model.mesh.elements)!=len(mesh.quads)+len(mesh.tris)+len(mesh.beams)+len(mesh.couplings):
        raise ValueError('consumer element identity collision')
    return model


def load_case(fixture,mesh,kind):
    from anysolver.boundary import LoadCase
    load=LoadCase('SG1 '+kind)
    nodal={n:np.zeros(6) for n in mesh.nodes}
    if kind=='pressure':
        for e,b in {**mesh.quads,**mesh.tris}.items():
            family='Q8' if len(b)==8 else 'T6'
            p,w=quadrature(len(b)==6)
            xyz=np.array([mesh.nodes[n] for n in b])
            ev=evaluate_mapping(xyz,family,p)
            forces=shape_values(family,p).T@((1000*w)[:,None]*ev.jacobian_vector)
            for n,f in zip(b,forces): nodal[n][:3]+=f
    elif kind=='member':
        n=mesh.node_of_vertex[fixture.member_end]
        nodal[n][:3]=100*fixture.member_axis
    else: raise ValueError(kind)
    for n,f in nodal.items():
        if np.any(f): load.add_nodal_load(n,f.tolist())
    return load,nodal


def probe(fixture,mesh,model,u):
    result={}
    for face,patch in fixture.patches.items():
        target=np.array((.5,.5)); candidates=[]
        for e in mesh.elements_of_face[face]:
            body=mesh.quads.get(e,mesh.tris.get(e)); family='Q8' if len(body)==8 else 'T6'
            uv=np.array([patch.uv(mesh.nodes[n]) for n in body])
            if np.any(target<uv.min(axis=0)-1e-8) or np.any(target>uv.max(axis=0)+1e-8): continue
            fit=least_squares(lambda p:shape_values(family,p)@uv-target,[0.,0.] if family=='Q8' else [1/3,1/3],gtol=1e-13,xtol=1e-13,ftol=1e-13)
            inside=(max(abs(fit.x))<=1+1e-8 if family=='Q8' else min(fit.x)>=-1e-8 and sum(fit.x)<=1+1e-8)
            if inside and np.linalg.norm(fit.fun)<=1e-9:
                values=np.array([u[model.mesh.nodes[n].dofs[:3]] for n in body])
                candidates.append(shape_values(family,fit.x)@values)
        if not candidates: raise ValueError(f'no containing element for fixed source probe {face}')
        if max(np.linalg.norm(p-candidates[0]) for p in candidates)>1e-8*max(1.,np.linalg.norm(candidates[0])):
            raise ValueError('discontinuous probe interpolation')
        result[str(face)]=candidates[0].tolist()
    if fixture.member_end is not None:
        n=mesh.node_of_vertex[fixture.member_end]
        result['member_tip']=u[model.mesh.nodes[n].dofs[:3]].tolist()
    return result


def solve(fixture,mesh,kind='pressure'):
    from anysolver import solve_linear,ResourceConfig
    from anysolver.matrix_assembly import assemble_stiffness_matrix
    from anysolver.assembly import build_constraint_transformation
    start=time.perf_counter(); model=import_model(fixture,mesh); imported=time.perf_counter()-start
    load,nodal=load_case(fixture,mesh,kind)
    start=time.perf_counter()
    u,info=solve_linear(model,load,resource_config=ResourceConfig(solver_threads=1,assembly_threads=1))
    elapsed=time.perf_counter()-start
    K,_=assemble_stiffness_matrix(model)
    F=np.zeros(len(u))
    for n,f in nodal.items(): F[model.mesh.nodes[n].dofs]=f
    _,_,T,_,_,constraint_info=build_constraint_transformation(K,F,model)
    residual=K@u-F
    free=float(np.linalg.norm(T.T@residual)/max(np.linalg.norm(T.T@F),1e-30))
    reaction_checks,force,moment=reaction_audit(model,mesh,residual,F,constraint_info)
    energy=float(.5*u@(K@u))
    constraints=float(info['constraint_postcheck']['max_relative_residual'])
    checks=list(reaction_checks)
    check(checks,'solver/converged',info['convergence_info']['status']=='converged')
    check(checks,'solver/finite',np.isfinite(u).all())
    check(checks,'solver/positive-energy',energy>0,value=energy)
    check(checks,'solver/free-residual',free<=LIMITS['free_residual'],value=free)
    check(checks,'solver/constraint-residual',constraints<=LIMITS['constraint_residual'],value=constraints)
    check(checks,'solver/force-equilibrium',force<=LIMITS['equilibrium'],value=force)
    check(checks,'solver/moment-equilibrium',moment<=LIMITS['equilibrium'],value=moment)
    return dict(checks=checks,energy=energy,probes=probe(fixture,mesh,model,u),dofs=len(u),
        formulation='legacy-Q8/T6 + QuadraticBeamElement + InterpolatedBeamShellMPCElement',
        timings=dict(solver_import=imported,solver_total=elapsed,
            assembly=info['phase_timings_seconds'].get('assembly_and_reduction'),
            solution=info['phase_timings_seconds'].get('factorization_and_solve')),
        thread_policy=info.get('thread_policy'),constraint_info=constraint_info,
        displacement={str(n):u[model.mesh.nodes[n].dofs].tolist() for n in mesh.nodes})


def reaction_audit(model,mesh,residual,F,constraint_info):
    """Separate supports from MPC forces; independently audit each MPC wrench."""
    checks=[]; internal=np.zeros_like(residual)
    for eid,c in mesh.couplings.items():
        actual=model.mesh.elements[eid].get_mpc_constraints(model.mesh)
        slave=model.mesh.nodes[c.beam_node].dofs
        e=np.asarray(c.eccentricity); skew=np.array(((0,-e[2],e[1]),(e[2],0,-e[0]),(-e[1],e[0],0)))
        expected={d:{} for d in slave}
        for n,w in zip(c.plate_nodes,c.weights):
            ds=model.mesh.nodes[n].dofs
            for i in range(3):
                expected[slave[i]][ds[i]]=float(w)
                expected[slave[i+3]][ds[i+3]]=float(w)
                for j in range(3): expected[slave[i]][ds[j+3]]=-float(w)*skew[i,j]
        valid=len(actual)==6 and {r['slave'] for r in actual}==set(slave)
        contribution=np.zeros_like(residual)
        for equation in actual:
            s=equation['slave']; masters=equation['masters']; wanted=expected.get(s,{})
            valid &= equation.get('value',0)==0 and all(abs(masters.get(k,0)-wanted.get(k,0))<=1e-12 for k in set(masters)|set(wanted))
            multiplier=residual[s]; contribution[s]+=multiplier
            for master,coefficient in masters.items(): contribution[master]-=coefficient*multiplier
        check(checks,f'MPC/{eid}/declared-transfer',valid)
        internal+=contribution
        wrench=np.zeros(6)
        for n,p in mesh.nodes.items():
            f=contribution[model.mesh.nodes[n].dofs]
            wrench[:3]+=f[:3]; wrench[3:]+=f[3:]+np.cross(p-mesh.nodes[c.beam_node],f[:3])
        scale=max(float(np.linalg.norm(residual[slave])),1e-30)
        check(checks,f'MPC/{eid}/internal-wrench',np.linalg.norm(wrench)<=LIMITS['equilibrium']*scale,value=float(np.linalg.norm(wrench)/scale))
    support=np.zeros_like(residual); fixed=constraint_info['fixed_dofs']
    support[fixed]=(residual-internal)[fixed]
    remainder=residual-internal-support
    check(checks,'solver/reaction-free-remainder',np.linalg.norm(remainder)<=LIMITS['free_residual']*max(np.linalg.norm(F),1e-30),value=float(np.linalg.norm(remainder)/max(np.linalg.norm(F),1e-30)))
    total=np.zeros(6); denominator_force=denominator_moment=0.
    origin=np.mean(list(mesh.nodes.values()),axis=0)
    diameter=max(1.,float(np.linalg.norm(np.ptp(np.array(list(mesh.nodes.values())),axis=0))))
    for n,p in mesh.nodes.items():
        f=F[model.mesh.nodes[n].dofs]; r=support[model.mesh.nodes[n].dofs]; arm=p-origin
        total[:3]+=f[:3]+r[:3]
        total[3:]+=f[3:]+r[3:]+np.cross(arm,f[:3]+r[:3])
        denominator_force+=np.linalg.norm(f[:3]); denominator_moment+=np.linalg.norm(f[3:])+np.linalg.norm(np.cross(arm,f[:3]))
    force=float(np.linalg.norm(total[:3])/max(denominator_force,1e-30))
    moment=float(np.linalg.norm(total[3:])/max(denominator_moment,denominator_force*diameter,1e-30))
    return checks,force,moment
