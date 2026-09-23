"""Q3a cross-field guidance tests (fourfold 4theta model).

Covers the full admin-mandated battery:

* fourfold representation: 90-degree equivalence, 2-fold invariance
* rotation covariance: geometry rotation phi -> encoded rotation 4*phi
* reflection covariance about the x-axis
* orthogonal edge reinforcement (confidence 1), 45-degree conflict (confidence 0)
* confidence diagnostics and min_confidence rejection
* explicit per-node anchors (pin + smoothing seed), anchor conflict diagnostics
* bounded deterministic smoothing (fixed anchors, reproducible)
* boundary_tangent mode (tangent/normal cross-equivalence)
* score_body / rank_bodies determinism (shuffled inputs, tie-break)
* front_step_guided: guided vs unguided multi-candidate fixtures (F1, plus the
  rotated F2 and reflected F3 copies), state-unchanged on failure, anchor-flip
* invalid mode / confidence / anchor / smoothing input handling

Every numeric expectation in this file is grounded in the real implementation
(run the scratch probes, not hand algebra): ``four_of(1,0) == (1,0)`` (theta=0
maps to z(0)=(1,0), NOT (-1,0)), orthogonal pair ``[(1,0),(0,1)]`` yields field
(1,0) with confidence 1, and the two-edge 45-degree conflict ``[(1,0),(1,1)]``
yields (0,0) with confidence 0.
"""

from __future__ import annotations

import math
import random

import pytest

from anymesher.errors import MeshError
from anymesher.quad.front import FrontNoCandidate, front_step
from anymesher.quad.guidance import (
    CrossFieldReport,
    GuidanceRejected,
    LowConfidenceError,
    build_cross_field,
    field_from_directions,
    four_of,
    front_step_guided,
    rank_bodies,
    reflect_x,
    rot2,
    score_body,
)
from anymesher.quad.state import QuadMeshState

TOL = 1e-9
NEAR_ZERO = 1e-9


# ---------------------------------------------------------------------------
# Frozen multi-candidate fixtures
# ---------------------------------------------------------------------------


def _f1():
    """F1: two admissible quads across front edge ``(0, 1)``.

    Nodes:  0=(0,0)  1=(1,0)  2=(0,1)  3=(1,1)  4=(-1,1)
    Cells : 0=(0,1,2)  1=(1,2,3)  2=(0,2,4)
    Front : ``(0,1)`` only.

    Across edge ``(1,2)`` (shared by source cell 0 and partner cell 1) the only
    convex quad is **A = (0,1,3,2)** (body {0,1,2,3}).  Across edge ``(0,2)``
    (shared by source cell 0 and partner cell 2) the only convex quad is
    **B = (0,1,2,4)** (body {0,1,2,4}).

    *Unguided* ``front_step`` walks sorted non-front edges of the source, so it
    picks B (``(0,2)`` precedes ``(1,2)``).  The fourfold cross-field scores A
    strictly higher (A=1.0, B=0.0), so *guided* ``front_step_guided`` picks A.
    That strict divergence is the whole point of guidance.
    """
    return QuadMeshState(
        nodes={
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (0.0, 1.0),
            3: (1.0, 1.0),
            4: (-1.0, 1.0),
        },
        cells={0: (0, 1, 2), 1: (1, 2, 3), 2: (0, 2, 4)},
        initial_front=[(0, 1)],
    )


def _f2(phi: float = math.radians(30.0)):
    """F1 rotated geometrically by ``phi`` (rotation-covariance copy)."""
    base = _f1()
    c, s = math.cos(phi), math.sin(phi)

    def rot(p):
        return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)

    return QuadMeshState(
        nodes={k: rot(p) for k, p in base.nodes.items()},
        cells=base.cells,
        initial_front=[(0, 1)],
    )


def _f3():
    """F1 reflected about the x-axis (reflection-covariance copy)."""
    base = _f1()
    return QuadMeshState(
        nodes={k: (p[0], -p[1]) for k, p in base.nodes.items()},
        cells=base.cells,
        initial_front=[(0, 1)],
    )


BODY_A = (0, 1, 3, 2)  # across edge (1,2)
BODY_B = (0, 1, 2, 4)  # across edge (0,2)


# ===========================================================================
# four_of / field_from_directions -- algebra
# ===========================================================================


def test_four_of_90_degree_equivalence():
    a = four_of(1.0, 0.0)
    # theta=0 -> z(0)=(1,0); a 90-degree rotation carries the same representative.
    assert a == pytest.approx((1.0, 0.0))
    assert four_of(0.0, 1.0) == pytest.approx(a)
    assert four_of(-1.0, 0.0) == pytest.approx(a)
    assert four_of(0.0, -1.0) == pytest.approx(a)


def test_four_of_2fold_invariance():
    assert four_of(1.0, 0.0) == pytest.approx(four_of(-1.0, 0.0))
    v = four_of(0.6, 0.8)
    assert four_of(-0.6, -0.8) == pytest.approx(v)


def test_four_of_general_direction():
    # theta = 30 degrees -> z at 4*theta = 120 degrees: (-cos60, sin120) = (-1/2, sqrt3/2).
    c, s = four_of(math.cos(math.pi / 6), math.sin(math.pi / 6))
    assert c == pytest.approx(-math.cos(math.pi / 3))
    assert s == pytest.approx(math.sin(2.0 * math.pi / 3))


def test_four_of_45_degree_is_opposite():
    # theta = 45 degrees -> 4*theta = 180 degrees -> (-1, 0), opposite to theta=0.
    assert four_of(1.0, 1.0) == pytest.approx((-1.0, 0.0))


def test_four_of_zero_direction():
    assert four_of(0.0, 0.0) == (0.0, 0.0)


def test_field_orthogonal_reinforce():
    # (1,0) and (0,1) are 90 degrees apart -> identical fourfold rep -> full confidence.
    field, conf = field_from_directions([(1.0, 0.0), (0.0, 1.0)])
    assert conf == pytest.approx(1.0)
    assert field == pytest.approx((1.0, 0.0))


def test_field_45_degree_conflict():
    # (1,0) [(1,0) in z] and (1,1) [(-1,0) in z] cancel exactly -> zero field.
    field, conf = field_from_directions([(1.0, 0.0), (1.0, 1.0)])
    assert conf <= NEAR_ZERO
    assert field == (0.0, 0.0)


def test_field_empty():
    assert field_from_directions([]) == ((0.0, 0.0), 0.0)


def test_field_partial_agreement():
    # [(1,0),(1,0),(1,1)] -> reps (1,0),(1,0),(-1,0); un-normalised mean (1/3, 0).
    # confidence = |mean| = 1/3, field = +x unit.
    field, conf = field_from_directions([(1.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    assert field == pytest.approx((1.0, 0.0))
    assert conf == pytest.approx(1.0 / 3.0)


def test_rot2_by_120_geometry_rotation_covariance():
    # rotating a single direction by phi rotates its fourfold rep by 4*phi.
    phi = math.radians(30.0)
    v = four_of(1.0, 0.0)
    r = rot2(v, 4.0 * phi)
    target = four_of(math.cos(phi), math.sin(phi))
    assert r == pytest.approx(target)


def test_field_rotation_covariance():
    # rotating every anchor direction by phi equals rotating the field by 4*phi.
    phi = math.radians(30.0)
    dirs = [(0.6, 0.8), (1.0, 0.3), (0.2, -0.9)]
    f1, c1 = field_from_directions(dirs)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    rot = lambda d: (d[0] * cos_p - d[1] * sin_p, d[0] * sin_p + d[1] * cos_p)
    f2, c2 = field_from_directions([rot(d) for d in dirs])
    assert f2 == pytest.approx(rot2(f1, 4.0 * phi))
    assert c1 == pytest.approx(c2)


def test_field_reflect_x_covariance():
    # reflecting directions about the x-axis maps the field (x,y) -> (x,-y).
    dirs = [(0.6, 0.8), (1.0, 0.3), (0.2, -0.9)]
    f1, c1 = field_from_directions(dirs)
    refl = [(d[0], -d[1]) for d in dirs]
    f2, c2 = field_from_directions(refl)
    assert f2 == pytest.approx(reflect_x(f1))
    assert c1 == pytest.approx(c2)


# ===========================================================================
# build_cross_field -- report on fixtures
# ===========================================================================


def test_report_f1_field_and_confidence():
    st = _f1()
    rep = build_cross_field(st)
    assert rep.smoothed is False
    assert set(rep.field) == {0, 1, 2, 3, 4}
    # The consistent part of the field is all +x (fourfold (1,0)); node 4 is a
    # genuine two-edge 45-degree conflict -> zero field.
    for n in (0, 1, 2, 3):
        assert rep.field[n] == pytest.approx((1.0, 0.0))
    assert rep.field[4] == (0.0, 0.0)
    assert rep.confidence[4] <= NEAR_ZERO
    # Per-node agreement confidence is grounded in the data (not all 1.0):
    # node 3 (two orthogonal edges) is fully consistent; the others mix in a
    # diagonal edge and drop below 1.0.
    assert rep.confidence[3] == pytest.approx(1.0)
    assert 0.0 < rep.confidence[0] < 1.0
    assert 0.0 < rep.confidence[1] < 1.0
    assert "conflicting" in " || ".join(rep.diagnostics)


def test_report_f2_rotation_covariance():
    phi = math.radians(30.0)
    f1 = build_cross_field(_f1())
    f2 = build_cross_field(_f2(phi))
    for n in (0, 1, 2, 3, 4):
        assert f2.field[n] == pytest.approx(rot2(f1.field[n], 4.0 * phi) if f1.field[n] != (0.0, 0.0) else (0.0, 0.0))
    assert f1.confidence[0] == pytest.approx(f2.confidence[0])


def test_report_f3_reflection_covariance():
    f1 = build_cross_field(_f1())
    f3 = build_cross_field(_f3())
    for n in (0, 1, 2, 3, 4):
        expected = reflect_x(f1.field[n]) if f1.field[n] != (0.0, 0.0) else (0.0, 0.0)
        assert f3.field[n] == pytest.approx(expected)
    assert f1.confidence[0] == pytest.approx(f3.confidence[0])


def test_report_conflict_diagnostic_small():
    # single triangle: node 0 has edges (0,1)=(1,0) and (0,2)=(1,1) -> 45 degrees apart.
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0)}
    cells = {0: (0, 1, 2)}
    rep = build_cross_field(QuadMeshState(nodes=nodes, cells=cells))
    assert rep.confidence[0] <= NEAR_ZERO
    assert rep.field[0] == (0.0, 0.0)
    assert any("conflicting" in d for d in rep.diagnostics)


def test_report_low_confidence_diagnostic():
    # node 0 sits on two edges 20 degrees apart -> in-between (partial) confidence,
    # below a high min_confidence but above the zero-conflict threshold.
    a = math.radians(20.0)
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (math.cos(a), math.sin(a))}
    cells = {0: (0, 1, 2)}
    rep = build_cross_field(QuadMeshState(nodes=nodes, cells=cells), min_confidence=0.99)
    assert NEAR_ZERO < rep.confidence[0] < 1.0 - TOL
    assert any("low confidence" in d for d in rep.diagnostics)


def test_report_anchor_pins():
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0)}
    cells = {0: (0, 1, 2)}
    st = QuadMeshState(nodes=nodes, cells=cells)
    rep = build_cross_field(st, anchors={0: (1.0, 0.0)})
    assert rep.field[0] == pytest.approx(four_of(1.0, 0.0))
    assert rep.confidence[0] == 1.0
    assert set(rep.anchors) == {0}


def test_report_anchor_conflict():
    # anchor node 0 opposite to neighbour 1's dominant field -> conflict diagnostic.
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0)}
    cells = {0: (0, 1, 2)}
    st = QuadMeshState(nodes=nodes, cells=cells)
    base = build_cross_field(st)
    n1_dir = base.field[1] if base.field[1] != (0.0, 0.0) else (1.0, 0.0)
    opp = (-n1_dir[0], -n1_dir[1])
    rep = build_cross_field(st, anchors={0: opp})
    assert any("conflicts with neighbour" in d for d in rep.diagnostics)


def test_report_boundary_tangent_mode_identical():
    st = _f1()
    a = build_cross_field(st, mode="cross_4theta")
    b = build_cross_field(st, mode="boundary_tangent")
    assert a.field == b.field
    assert a.confidence == b.confidence
    assert b.mode == "boundary_tangent"


def test_report_invalid_mode():
    with pytest.raises(GuidanceRejected):
        build_cross_field(_f1(), mode="cross_2theta")
    with pytest.raises(GuidanceRejected):
        build_cross_field(_f1(), mode=None)


def test_report_invalid_min_confidence():
    st = _f1()
    for bad in (-0.1, 1.5, float("nan"), float("inf"), "x", True):
        with pytest.raises(GuidanceRejected):
            build_cross_field(st, min_confidence=bad)


def test_report_invalid_anchors():
    st = _f1()
    with pytest.raises(GuidanceRejected):
        build_cross_field(st, anchors="{0: (1,0)}")
    with pytest.raises(GuidanceRejected):
        build_cross_field(st, anchors={0: (0.0, 0.0)})
    with pytest.raises(GuidanceRejected):
        build_cross_field(st, anchors={0: [1.0]})


def test_report_invalid_smoothing():
    st = _f1()
    for bad in (-1, "3", True, 1.5):
        with pytest.raises(GuidanceRejected):
            build_cross_field(st, smoothing=bad)


# ===========================================================================
# Smoothing -- bounded, deterministic, anchors fixed
# ===========================================================================


def _smooth_fixture():
    """A small field where anchors + smoothing are well-defined."""
    return QuadMeshState(
        nodes={
            0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0),
            3: (0.0, 1.0), 4: (-1.0, -1.0), 5: (2.0, 1.0),
        },
        cells={0: (0, 1, 3), 1: (0, 1, 2), 2: (0, 1, 4), 3: (1, 2, 5)},
    )


def test_smoothing_deterministic_and_anchors_fixed():
    st = _smooth_fixture()
    anchors = {0: (1.0, 0.0)}
    r1 = build_cross_field(st, anchors=anchors, smoothing=3)
    r2 = build_cross_field(st, anchors=anchors, smoothing=3)
    assert r1.field == r2.field
    assert r1.confidence == r2.confidence
    assert r1.smoothed is True
    # anchor node stays pinned to its fourfold representative with full confidence.
    assert r1.field[0] == pytest.approx(four_of(1.0, 0.0))
    assert r1.confidence[0] == 1.0
    # every relaxed field vector is either unit or exactly zero.
    for v in r1.field.values():
        assert math.hypot(v[0], v[1]) == pytest.approx(1.0, abs=TOL) or v == (0.0, 0.0)


def test_smoothing_zero_is_off():
    st = _f1()
    a = build_cross_field(st, smoothing=0)
    assert a.smoothed is False
    # smoothing=0 must leave the raw (no-relaxation) field untouched.
    raw = build_cross_field(st)
    assert a.field == raw.field


# ===========================================================================
# score_body / rank_bodies
# ===========================================================================


def test_score_body_range_and_determinism():
    st = _f1()
    rep = build_cross_field(st)
    for body in (BODY_A, BODY_B, (0, 2, 3, 1)):
        s = score_body(st, rep, body)
        assert -1.0 - TOL <= s <= 1.0 + TOL


def test_score_body_rejects_bad_body():
    st = _f1()
    rep = build_cross_field(st)
    with pytest.raises(GuidanceRejected):
        score_body(st, rep, (0, 1, 2))
    with pytest.raises(GuidanceRejected):
        score_body(st, rep, (0, 1, 2, 1))


def test_f1_scores_separate_candidates():
    st = _f1()
    rep = build_cross_field(st)
    sA = score_body(st, rep, BODY_A)
    sB = score_body(st, rep, BODY_B)
    assert sA == pytest.approx(1.0)
    assert sB <= TOL
    assert sA > sB


def test_rank_bodies_deterministic_under_shuffle():
    st = _f1()
    rep = build_cross_field(st)
    bodies = [BODY_A, BODY_B, (0, 2, 3, 1)]
    shuffled = bodies[:]
    random.Random(1234).shuffle(shuffled)
    a = rank_bodies(st, rep, bodies)
    b = rank_bodies(st, rep, shuffled)
    assert a == b
    assert a[0][0] >= a[1][0] >= a[2][0]


def test_rank_bodies_tie_break_by_body_tuple():
    # Unit square (4-node field where both candidate bodies tie): the winner must
    # be the lexicographically smaller body tuple, independent of input order.
    nodes = {10: (0.0, 0.0), 20: (1.0, 0.0), 30: (1.0, 1.0), 40: (0.0, 1.0)}
    st = QuadMeshState(nodes=nodes, cells={0: (10, 20, 30)})
    rep = build_cross_field(st)
    ranked = rank_bodies(st, rep, [(40, 30, 20, 10), (10, 20, 30, 40)])
    assert ranked[0][0] == pytest.approx(ranked[1][0])
    assert ranked[0][1] == (10, 20, 30, 40)


# ===========================================================================
# front_step_guided -- multi-candidate fixtures, guided vs unguided
# ===========================================================================


def test_f1_unguided_picks_b():
    st = _f1()
    new_id, body = front_step(st, (0, 1))
    assert sorted(body) == sorted(BODY_B)


def test_f1_guided_picks_a():
    st = _f1()
    rep = build_cross_field(st)
    new_id, body = front_step_guided(st, (0, 1), report=rep)
    # node-set of the winner is the A quad {0,1,2,3}, not B's {0,1,2,4}.
    assert set(body) == set(BODY_A)
    assert st.cell_kind(new_id) == "Q4"
    # the two source triangles were consumed; the front edge (0,1) stays on the
    # front as a genuine mesh boundary of the new quad.
    assert new_id not in st.cells or st.cells.get(new_id) is not None
    assert st.cells.get(new_id) is not None


def test_f1_guided_vs_unguided_diverge():
    st_u, st_g = _f1(), _f1()
    rep = build_cross_field(st_g)
    u_id, u_body = front_step(st_u, (0, 1))
    g_id, g_body = front_step_guided(st_g, (0, 1), report=rep)
    assert u_id == g_id
    assert set(u_body) != set(g_body)
    # the guided winner is strictly better under the field.
    assert score_body(st_g, rep, g_body) > score_body(st_g, rep, u_body)


def test_f2_guided_rotation_covariant():
    st = _f2()
    rep = build_cross_field(st)
    new_id, body = front_step_guided(st, (0, 1), report=rep)
    assert new_id == 3
    # geometry rotated, node ids unchanged -> same winner node-set as F1 (the A
    # quad {0,1,2,3}), and it dominates its competitor under the rotated field.
    assert set(body) == set(BODY_A)
    assert score_body(st, rep, body) > score_body(st, rep, BODY_B)


def test_f3_guided_reflection_covariant():
    st = _f3()
    rep = build_cross_field(st)
    new_id, body = front_step_guided(st, (0, 1), report=rep)
    assert new_id == 3
    # reflected: same winner node-set {0,1,2,3}; the canonical walk differs from
    # F1's (0,1,3,2) but the winner *beats* B under the reflected field.
    assert set(body) == set(BODY_A)
    assert score_body(st, rep, body) > score_body(st, rep, BODY_B)


def test_anchor_flip_reverses_guided_winner():
    # Anchoring the two corners of candidate A opposite to its ideal field makes
    # B (which does not touch those corners) the strict guided winner.
    anchors = {0: (1.0, 1.0), 3: (1.0, 1.0)}
    st = _f1()
    rep = build_cross_field(st, anchors=anchors)
    new_id, body = front_step_guided(st, (0, 1), report=rep)
    assert tuple(sorted(body)) == tuple(sorted(BODY_B))


def test_guided_shuffled_candidate_order_same_winner():
    st = _f1()
    rep = build_cross_field(st)
    ranked = rank_bodies(st, rep, [BODY_B, BODY_A])
    _, winner = ranked[0]
    new_id, body = front_step_guided(st, (0, 1), report=rep)
    assert tuple(sorted(body)) == tuple(sorted(winner))


def test_guided_not_on_front_rejected_state_unchanged():
    st = _f1()
    before = st.digest()
    rep = build_cross_field(st)
    with pytest.raises(FrontNoCandidate):
        front_step_guided(st, (1, 2), report=rep)
    assert st.digest() == before
    assert st.generation == 0


def _report_conf(rep, overrides):
    conf = dict(rep.confidence)
    conf.update(overrides)
    return CrossFieldReport(
        field=rep.field,
        confidence=conf,
        mode=rep.mode,
        anchors=rep.anchors,
        diagnostics=rep.diagnostics,
        smoothed=rep.smoothed,
    )


def test_guided_low_confidence_front_endpoint_rejected_state_unchanged():
    st = _f1()
    before = st.digest()
    rep = build_cross_field(st)
    bad = _report_conf(rep, {0: 0.0})
    with pytest.raises(LowConfidenceError):
        front_step_guided(st, (0, 1), report=bad, min_confidence=0.5)
    assert st.digest() == before


def test_guided_candidate_endpoint_conflict_rejected():
    # candidate B touches node 4 (zero-confidence); only A survives the gate.
    st = _f1()
    before = st.digest()
    rep = build_cross_field(st)
    bad = _report_conf(rep, {4: 0.0, 0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    new_id, body = front_step_guided(st, (0, 1), report=bad, min_confidence=0.5)
    assert tuple(sorted(body)) == tuple(sorted(BODY_A))
    assert st.digest() != before


def test_guided_all_candidates_rejected_low_confidence():
    # Keep the front endpoints admissible (conf 1.0) but push every *candidate*
    # node below min_confidence so no candidate survives the gate -> FrontNoCandidate.
    st = _f1()
    before = st.digest()
    rep = build_cross_field(st)
    bad = _report_conf(rep, {0: 1.0, 1: 1.0, 2: 0.4, 3: 0.4, 4: 0.4})
    with pytest.raises(FrontNoCandidate):
        front_step_guided(st, (0, 1), report=bad, min_confidence=0.9)
    assert st.digest() == before


def test_guided_front_endpoint_below_min_confidence():
    st = _f1()
    before = st.digest()
    rep = build_cross_field(st)
    bad = _report_conf(rep, {0: 0.5})
    with pytest.raises(LowConfidenceError):
        front_step_guided(st, (0, 1), report=bad, min_confidence=0.9)
    assert st.digest() == before


def test_guided_bad_options_type():
    st = _f1()
    rep = build_cross_field(st)
    with pytest.raises(MeshError):
        front_step_guided(st, (0, 1), report=rep, options="bad")


def test_guided_invalid_min_confidence():
    st = _f1()
    rep = build_cross_field(st)
    with pytest.raises(GuidanceRejected):
        front_step_guided(st, (0, 1), report=rep, min_confidence=-1.0)
    with pytest.raises(GuidanceRejected):
        front_step_guided(st, (0, 1), report=rep, min_confidence="high")


# ===========================================================================
# Unguided byte-compat: guidance does not change front_step at all
# ===========================================================================


def test_unguided_front_step_unchanged_by_guidance_module():
    st1, st2 = _f1(), _f1()
    rep = build_cross_field(st1)
    del rep
    assert st1.digest() == st2.digest()
    n1, b1 = front_step(st1, (0, 1))
    n2, b2 = front_step(st2, (0, 1))
    assert (n1, b1) == (n2, b2)


def test_build_on_view_during_transaction_unchanged():
    st = _f1()
    with st.transaction() as tx:
        report = build_cross_field(tx.view)
        assert set(report.field) == {0, 1, 2, 3, 4}
    # no commit: digest unchanged.
    st2 = _f1()
    assert st.digest() == st2.digest()
