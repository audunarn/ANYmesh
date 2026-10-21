"""Fixed-boundary local transition quality and transactional publication."""
import pytest
import numpy as np

from anygeometry import GeometryModel
from anymesher.errors import MeshError
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.optimize import optimize_quad_state
from anymesher.quad.state import QuadMeshState
from anymesher.quad.transition_smoothing import smooth_planar_transitions
import anymesher.quad.transition_smoothing as smoothing


def domain():
    geometry = GeometryModel()
    vertices = geometry.add_points(((0., 0., 0.), (2., 0., 0.),
                                    (2., 2., 0.), (0., 2., 0.)))
    face = geometry.add_plate(vertices)
    return PlanarQuadDomain.from_geometry(geometry, face)


def distorted_quads():
    nodes = {0:(0.,0.),1:(1.,0.),2:(2.,0.),3:(2.,1.),4:(2.,2.),
             5:(1.,2.),6:(0.,2.),7:(0.,1.),8:(1.65,1.2)}
    cells = {0:(0,1,8,7),1:(1,2,3,8),2:(8,3,4,5),3:(7,8,5,6)}
    return QuadMeshState(nodes,cells,protected_nodes=range(8))


def growth_transition():
    nodes = {0:(-.7,0.),1:(0.,0.),2:(0.,.5),3:(-.7,.5),
             4:(-.23333333333333334,.25),5:(.7,0.),6:(.7,.5)}
    cells = {0:(0,1,4),1:(1,2,4),2:(2,3,4),3:(3,0,4),4:(1,5,6,2)}
    return QuadMeshState(nodes,cells,protected_nodes=(0,1,2,3,5,6))


@pytest.mark.parametrize('factory', [distorted_quads, growth_transition])
def test_mixed_and_boundary_neighbor_cavities_improve_without_topology_change(factory):
    state = factory()
    before = state.snapshot()
    result = smooth_planar_transitions(state, domain(), budget=8)
    assert result['status'] == 'APPLIED'
    assert result['final_excess'] < result['initial_excess']
    assert result['evaluations'] <= result['evaluation_budget'] == 4096
    after = state.snapshot()
    for key in ('cells','cell_kinds','edge_cells','front','front_bits','protected_nodes','protected_edges'):
        assert after[key] == before[key]
    for node in before['protected_nodes']:
        assert after['nodes'][node] == before['nodes'][node]
    clone = factory()
    assert smooth_planar_transitions(clone, domain(), budget=8) == result
    assert clone.snapshot() == after


@pytest.mark.parametrize('mode', ['zero', 'custom_domain', 'protected'])
def test_disabled_unsupported_and_no_free_node_are_exact_noops(mode):
    state = distorted_quads()
    owner = domain()
    if mode == 'custom_domain':
        owner = object()
    if mode == 'protected':
        snap = state.snapshot()
        state = QuadMeshState(snap['nodes'],snap['cells'],protected_nodes=snap['nodes'])
    before = state.snapshot()
    smooth_planar_transitions(state, owner, budget=0 if mode == 'zero' else 8)
    assert state.snapshot() == before


@pytest.mark.parametrize('phase', ['search', 'commit'])
def test_cancel_is_atomic_and_preserves_exception_identity(phase):
    state = distorted_quads()
    before = state.snapshot()
    marker = MeshError('cancel transition')
    def cancel(stage):
        if stage.endswith(phase):
            raise marker
    with pytest.raises(MeshError) as caught:
        smooth_planar_transitions(state, domain(), budget=8, cancellation_check=cancel)
    assert caught.value is marker
    assert state.snapshot() == before


def test_authoritative_neighbor_regression_rejects_complete_search(monkeypatch):
    state = distorted_quads()
    before = state.snapshot()
    original = smoothing._common_evidence
    calls = []
    def evidence(*args):
        result = original(*args)
        if calls:
            result[('cell',0)] = tuple(value+1. for value in result[('cell',0)])
        calls.append(True)
        return result
    monkeypatch.setattr(smoothing, '_common_evidence', evidence)
    report = smooth_planar_transitions(state, domain(), budget=8)
    assert report['status'] == 'REJECTED_COMMON_QUALITY'
    assert state.snapshot() == before


def test_optimizer_optin_keeps_worker_counts_and_zero_budget(monkeypatch):
    monkeypatch.delenv('ANYMESH_Q5_DISABLE_WORKER', raising=False)
    owner = domain()
    legacy = distorted_quads()
    before = legacy.snapshot()
    report = optimize_quad_state(legacy,target_size=1.,max_local_optimizations=8,domain=owner)
    assert report.transition_quality is None and legacy.snapshot() == before
    candidate = distorted_quads()
    report = optimize_quad_state(candidate,target_size=1.,max_local_optimizations=8,
                                 domain=owner,linear_transition_quality=True)
    assert report.worker_calls == 0
    assert report.transition_quality['status'] == 'APPLIED'
    disabled = distorted_quads()
    before = disabled.snapshot()
    report = optimize_quad_state(disabled,target_size=1.,max_local_optimizations=0,
                                 domain=owner,linear_transition_quality=True)
    assert report.transition_quality['status'] == 'DISABLED'
    assert disabled.snapshot() == before


def test_explicit_worker_disable_is_also_transition_disable(monkeypatch):
    monkeypatch.setenv('ANYMESH_Q5_DISABLE_WORKER','1')
    state = distorted_quads()
    before = state.snapshot()
    report = optimize_quad_state(state,target_size=1.,max_local_optimizations=8,
                                 domain=domain(),linear_transition_quality=True)
    assert report.transition_quality is None
    assert state.snapshot() == before


def test_passing_face_checks_cancellation_before_setup():
    state = QuadMeshState({0:(0.,0.),1:(1.,0.),2:(1.,1.),3:(0.,1.)},
                          {0:(0,1,2,3)},protected_nodes=range(4))
    before = state.snapshot()
    marker = MeshError('cancel passing face')
    calls = []
    def cancel(stage):
        calls.append(stage)
        raise marker
    with pytest.raises(MeshError) as caught:
        smooth_planar_transitions(state,domain(),budget=8,cancellation_check=cancel)
    assert caught.value is marker
    assert calls == ['quad-first:planar-transition-setup']
    assert state.snapshot() == before


def test_assembled_guard_detects_cross_face_growth_regression_without_mutation():
    from anymesher.mesh import Mesh
    from anymesher.quad.transition_smoothing import adjudicate_assembled_transition
    # Each face alone has acceptable shape. Shortening the left cell while
    # retaining its shared edge makes only the cross-face growth pair fail.
    nodes = {0:(0.,0.,0.),1:(1.,0.,0.),2:(1.,1.,0.),3:(.3,.8,0.),
             4:(3.,0.,0.),5:(3.,1.,0.)}
    mesh = Mesh(nodes={n:np.array(p) for n,p in nodes.items()},
                quads={0:(0,1,2,3),1:(1,4,5,2)},elements_of_face={1:[0],2:[1]},order='linear')
    saved = {n:p.copy() for n,p in mesh.nodes.items()}
    result = adjudicate_assembled_transition(mesh,{3:np.array((0.,1.,0.))})
    assert not result['accepted'] and result['proposed_excess'] > result['initial_excess']
    assert all(np.array_equal(mesh.nodes[n],p) for n,p in saved.items())


def test_assembled_guard_accepts_complete_improvement():
    from anymesher.mesh import Mesh
    from anymesher.quad.transition_smoothing import adjudicate_assembled_transition
    state = distorted_quads()
    owner = domain()
    original = {n:np.array(owner.lift(state.position(n))) for n in state.nodes}
    report = smooth_planar_transitions(state,owner,budget=8)
    assert report['status'] == 'APPLIED'
    mesh = Mesh(nodes={n:np.array(owner.lift(state.position(n))) for n in state.nodes},
                quads={cid:state.cell(cid) for cid in state.cells},
                elements_of_face={1:[0,1],2:[2,3]},order='linear')
    result = adjudicate_assembled_transition(mesh,original)
    assert result['accepted'] and result['proposed_excess'] < result['initial_excess']


def test_public_full_shell_rollback_refreshes_validation_and_preserves_q5(monkeypatch):
    import anymesher.quad.optimize as optimizer
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.quad.options import QuadMeshingOptions
    from anymesher.refinement import refine_at
    monkeypatch.delenv('ANYMESH_Q5_DISABLE_WORKER', raising=False)
    monkeypatch.setattr(optimizer,'_optimize_quad_state_q5',
        lambda *args, **kwargs: optimizer._report('NO_ELIGIBLE',eligible=0,budget=8))
    geometry = GeometryModel()
    vertices = geometry.add_points(((0.,0.,0.),(8.,0.,0.),(8.,4.,0.),(0.,4.,0.)))
    geometry.add_plate(vertices)
    common = dict(target_size=.5,strategy='native',order='linear',layout_policy='adaptive',
                  refinements=[refine_at((2.,2.,0.),.25,radius=.75)])
    reference = generate_hybrid_mesh_result(geometry,quad_options=QuadMeshingOptions(max_local_optimizations=0),**common)
    audited = []
    def reject(mesh, positions, cancellation_check=None):
        audited.append(bool(positions))
        return dict(accepted=False,status='ROLLED_BACK',initial_excess=1.,proposed_excess=2.,
                    cross_domain_pairs_included=True)
    monkeypatch.setattr(smoothing,'adjudicate_assembled_transition',reject)
    actual = generate_hybrid_mesh_result(geometry,quad_options=QuadMeshingOptions(max_local_optimizations=8),**common)
    assert audited == [True]
    assert actual.mesh.quads == reference.mesh.quads and actual.mesh.tris == reference.mesh.tris
    assert all(np.array_equal(point,actual.mesh.nodes[n]) for n,point in reference.mesh.nodes.items())
    assert actual.mesh.hybrid_diagnostics['validation'] == reference.mesh.hybrid_diagnostics['validation']
    reports = actual.mesh.hybrid_diagnostics['q5']['faces'].values()
    assert any(item.get('transition_quality',{}).get('status') == 'ROLLED_BACK_FULL_SHELL' for item in reports)
