"""Quad cross-field guidance for the advancing front (Q3a).

This layer is *additive and opt-in*.  It computes a deterministic, bounded
cross-field from a :class:`~.state.QuadMeshState` and uses it **only to rank**
the admissible candidates already produced by :mod:`anymesher.quad.front`.
All geometry, admissibility, protected-feature, rollback and transaction rules
stay authoritative in :mod:`anymesher.quad.front`; guidance never relaxes a rule
that geometry rejects, and :func:`anymesher.quad.front.front_step` (unguided)
is byte-compatible with pre-Q3 behaviour.

Cross-field theory (the Q0 fourfold contract)
---------------------------------------------
A quad cross-field is a *line* field in the plane defined up to 90 degrees:
a quad's two edge directions are orthogonal, and swapping them is the same
field.  The faithful plane representation is the fourfold map

    z(theta) = (cos 4 theta, sin 4 theta)

so ``theta`` is periodic with ``pi/2`` (90 degrees).  Consequences used by the
tests:

* **90-degree equivalence:** a direction and its 90-degree rotation carry the
  same ``z`` (4*(theta+pi/2) = 4 theta + 2pi).
* **Rotation covariance:** rotating the geometry by ``phi`` rotates the
  encoded field by ``4*phi`` in the ``z``-plane.  A 90-degree geometry
  rotation therefore leaves the encoded field invariant.
* **Reflection covariance:** reflecting the geometry about the x-axis maps
  ``theta -> -theta`` and hence ``z -> (cos4theta, -sin4theta)``.

The per-node field is the normalised mean of the incident edges' ``z``
representatives; the *confidence* is the magnitude of that mean (1 for
fully consistent anchors, -> 0 for conflicting anchors, e.g. two edges 45
degrees apart).  An explicit per-node direction **anchor** pinces a node and
also seeds the bounded smoothing below.

Boundary / boundary-tangent semantics: in ``boundary_tangent`` mode the
boundary (front) edges anchor on their *tangent* direction; under the fourfold
representation tangent and normal are cross-equivalent (both map to the same
``z``), so the anchor value is identical either way — the mode documents the
tangent interpretation while staying mathematically consistent.

Bounded sparse smoothing
------------------------
The encoded ``z`` vectors may optionally be relaxed over mesh adjacency for a
fixed, bounded number of iterations.  Anchored nodes are fixed; the relaxation
is deterministic (sorted node order), sparse (node-neighbourhood only), and is
**not** a global parameterisation solver.

Candidate scoring
-----------------
:func:`score_body` compares each *boundary edge* of a canonical convex quad
body against the local node fields and averages deterministically.  Because two
adjacent orthogonal boundary edges of a quad share the same fourfold
representative, this is the intended alignment score: a body whose boundary
edges agree with the field scores higher.  :func:`front_step_guided` ranks the
*already admissible* candidates by this score and commits the winner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..errors import MeshError
from .front import (
    FrontNoCandidate,
    FrontRejected,
    body_edges,
    candidate_partners,
    classify,
    edge_key,
    find_source_cell,
)
from .options import QuadMeshingOptions
from .state import QuadMeshState, View

__all__ = [
    "GuidanceRejected",
    "LowConfidenceError",
    "CrossFieldReport",
    "NEAR_ZERO",
    "four_of",
    "field_from_directions",
    "build_cross_field",
    "rot2",
    "reflect_x",
    "score_body",
    "rank_bodies",
    "front_step_guided",
]

# Tolerance for near-zero (conflict) detection.
NEAR_ZERO = 1e-9


class GuidanceRejected(MeshError):
    """A specific guidance rule rejected the configuration (typed, non-fatal)."""


class LowConfidenceError(GuidanceRejected):
    """The cross-field confidence at a candidate endpoint is below ``min_confidence``."""


@dataclass(frozen=True)
class CrossFieldReport:
    """Per-node fourfold field, confidence, anchors and diagnostics (immutable)."""

    field: Mapping[int, tuple[float, float]]
    confidence: Mapping[int, float]
    mode: str
    anchors: Mapping[int, tuple[float, float]]
    diagnostics: tuple[str, ...] = field(default=())
    smoothed: bool = False


# ---------------------------------------------------------------------------
# Pure vector helpers (cross-field algebra; no mesh state needed)
# ---------------------------------------------------------------------------


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    l = math.hypot(v[0], v[1])
    if l <= NEAR_ZERO:
        return (0.0, 0.0)
    return (v[0] / l, v[1] / l)


def four_of(dx: float, dy: float) -> tuple[float, float]:
    """Fourfold representation ``z = (cos 4theta, sin 4theta)`` of a direction.

    Takes any (non-zero) direction ``(dx, dy)``; the result is a unit vector in
    the ``z``-plane.  The map is sign-invariant to 90 degrees:
    ``four_of`` of a direction and of its 90-degree rotation are identical, and
    it is a 2-fold-invariant (``theta`` vs ``theta + pi`` also identical).  A
    zero direction returns the zero vector.
    """
    c2 = dx * dx - dy * dy
    s2 = 2.0 * dx * dy
    n = math.hypot(c2, s2)
    if n <= NEAR_ZERO:
        return (0.0, 0.0)
    c2 /= n
    s2 /= n
    return (c2 * c2 - s2 * s2, 2.0 * c2 * s2)


def field_from_directions(
    directions: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], float]:
    """Mean fourfold field + confidence of a set of directions at one node.

    Returns ``(field_vector, confidence)`` where ``field_vector`` is the
    normalised mean of the fourfold representatives and ``confidence`` is the
    magnitude of the (un-normalised) mean.  ``confidence == 1.0`` means the
    anchors fully agree; ``confidence == 0.0`` means they cancel (e.g. two
    directions 45 degrees apart).  An empty input yields ``((0,0), 0.0)``.
    """
    total_x = total_y = 0.0
    k = 0
    for d in directions:
        c, s = four_of(float(d[0]), float(d[1]))
        total_x += c
        total_y += s
        k += 1
    if k == 0:
        return ((0.0, 0.0), 0.0)
    mx = total_x / k
    my = total_y / k
    conf = math.hypot(mx, my)
    return (_unit((mx, my)) if conf > NEAR_ZERO else (0.0, 0.0), conf)


def rot2(v: tuple[float, float], angle: float) -> tuple[float, float]:
    """Rotate an encoded ``z``-plane vector by ``angle`` (encoded radians).

    A geometry rotation of ``phi`` maps the field by ``rot2(field, 4*phi)``.
    """
    c, s = math.cos(angle), math.sin(angle)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def reflect_x(v: tuple[float, float]) -> tuple[float, float]:
    """Reflect an encoded ``z``-plane vector about the x-axis: ``(x, y) -> (x, -y)``."""
    return (v[0], -v[1])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_mode(mode: Any) -> None:
    allowed = {"cross_4theta", "boundary_tangent"}
    if mode not in allowed:
        raise GuidanceRejected(f"mode {mode!r} must be one of {sorted(allowed)!r}")


def _validate_real(value: Any, name: str, *, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuidanceRejected(f"{name} must be a real number")
    v = float(value)
    if math.isnan(v) or math.isinf(v) or v < lo or v > hi:
        raise GuidanceRejected(f"{name} {value!r} must be in [{lo}, {hi}]")
    return v


def _validate_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuidanceRejected(f"{name} must be an int")
    v = int(value)
    if v < 0:
        raise GuidanceRejected(f"{name} {value!r} must be >= 0")
    return v


def _validate_anchors(anchors: Any) -> dict[int, tuple[float, float]]:
    if anchors is None:
        return {}
    if not isinstance(anchors, Mapping):
        raise GuidanceRejected("anchors must be a mapping of node id -> (dx, dy)")
    out: dict[int, tuple[float, float]] = {}
    for k, v in anchors.items():
        nid = int(k)
        if isinstance(nid, bool):
            raise GuidanceRejected("anchor node id must be an int")
        try:
            dx, dy = float(v[0]), float(v[1])
        except (TypeError, IndexError, ValueError) as exc:
            raise GuidanceRejected(f"anchor for node {nid} must be a 2-vector") from exc
        if math.hypot(dx, dy) <= NEAR_ZERO:
            raise GuidanceRejected(f"anchor for node {nid} must be non-zero")
        out[nid] = (dx, dy)
    return out


# ---------------------------------------------------------------------------
# Cross-field construction
# ---------------------------------------------------------------------------


def _edge_rep_4(
    positions: Mapping[int, Sequence[float]], a: int, b: int
) -> tuple[float, float]:
    pa = positions[a]
    pb = positions[b]
    return four_of(float(pb[0]) - float(pa[0]), float(pb[1]) - float(pa[1]))


def _mean_rep(
    reps: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], float]:
    """Normalised mean of fourfold representatives + confidence (magnitude)."""
    total_x = total_y = 0.0
    k = 0
    for c, s in reps:
        total_x += c
        total_y += s
        k += 1
    if k == 0:
        return ((0.0, 0.0), 0.0)
    mx = total_x / k
    my = total_y / k
    conf = math.hypot(mx, my)
    return (_unit((mx, my)) if conf > NEAR_ZERO else (0.0, 0.0), conf)


def _smooth(
    field: dict[int, tuple[float, float]],
    adjacency: dict[int, list[int]],
    anchors: dict[int, tuple[float, float]],
    iterations: int,
) -> dict[int, tuple[float, float]]:
    """Bounded deterministic sparse Jacobi relaxation of encoded z-vectors."""
    cur = dict(field)
    if iterations <= 0:
        return cur
    for _ in range(iterations):
        nxt = dict(cur)
        for n in sorted(cur):
            if n in anchors:
                continue  # fixed anchor
            nbs = adjacency.get(n) or []
            if not nbs:
                continue
            sx = sum(cur[m][0] for m in nbs) / len(nbs)
            sy = sum(cur[m][1] for m in nbs) / len(nbs)
            u = _unit((sx, sy))
            if math.hypot(u[0], u[1]) > NEAR_ZERO:
                nxt[n] = u
        cur = nxt
    return cur


def build_cross_field(
    state: QuadMeshState | View,
    *,
    mode: str = "cross_4theta",
    min_confidence: float = 0.0,
    anchors: Mapping[int, Sequence[float]] | None = None,
    smoothing: int = 0,
) -> CrossFieldReport:
    """Build the per-node fourfold cross-field report on ``state`` (or ``View``).

    Deterministic in node-id order.  Reads only node positions and resident
    edge incidence — never writes.  Parameters:

    * ``mode`` — ``cross_4theta`` or ``boundary_tangent`` (see module docstring).
    * ``min_confidence`` — nodes (and later candidate endpoints) below this
      confidence are diagnostic; raising this at step time rejects.
    * ``anchors`` — explicit per-node direction anchors (pin + smoothing seed).
    * ``smoothing`` — bounded number of sparse Jacobi relaxation passes (0 = off).

    Diagnostics flag conflicting (zero) anchors, low-confidence nodes and
    explicit anchor conflicts with the local data.
    """
    _validate_mode(mode)
    min_conf = _validate_real(
        min_confidence, "min_confidence", lo=0.0, hi=1.0
    )
    anch = _validate_anchors(anchors)
    smoo = _validate_int(smoothing, "smoothing")

    node_ids = sorted(state.node_ids)
    positions = state.nodes

    raw: dict[int, tuple[float, float]] = {}
    conf: dict[int, float] = {}
    diagnostics: list[str] = []

    for n in node_ids:
        if n in anch:
            raw[n] = four_of(*anch[n])
            conf[n] = 1.0
            continue
        edges = state.neighbors_at(n)
        if not edges:
            continue
        # In every mode an edge anchors on its own direction.  In the fourfold
        # representation tangent and normal are cross-equivalent (both map to
        # the same z), so ``boundary_tangent`` keeps the identical
        # representative while documenting the tangent interpretation (Q0).
        reps = [_edge_rep_4(positions, a, b) for a, b in edges]
        vec, c = _mean_rep(reps)
        raw[n] = vec
        conf[n] = c
        if c <= NEAR_ZERO:
            diagnostics.append(f"node {n}: conflicting anchors (zero-mean, k={len(reps)})")
        elif c < min_conf:
            diagnostics.append(f"node {n}: low confidence {c:.6f}")

    # Bounded sparse smoothing over node adjacency (anchored nodes fixed).
    adjacency: dict[int, list[int]] = {n: [] for n in node_ids}
    for n in node_ids:
        for a, b in state.neighbors_at(n):
            other = a if b == n else b
            adjacency[n].append(other)

    smoothed = smoo > 0 and raw
    raw = _smooth(raw, adjacency, anch, smoo)

    # Confidence is the *data-agreement* of the local anchors (the magnitude of
    # their un-normalised fourfold mean), computed above.  Smoothing may rotate
    # a node's field direction but never changes how consistent those raw anchors
    # were, so the per-node confidence is preserved (anchored nodes are pinned at
    # 1.0 in the loop above).  We deliberately do NOT re-derive it here: a field
    # vector is already unit, so ``hypot`` would collapse every value to 0 or 1.

    # Anchor-conflict diagnostics: an anchored node adjacent to a node whose
    # dominant field opposes the anchor's fourfold direction.
    for n in sorted(anch):
        a_rep = _unit((anch[n][0], anch[n][1]))
        for other in adjacency.get(n, []):
            if other not in raw:
                continue
            o_rep = _unit(raw[other])
            if a_rep[0] * o_rep[0] + a_rep[1] * o_rep[1] <= -0.999999:
                diagnostics.append(
                    f"node {n}: explicit anchor conflicts with neighbour {other}"
                )

    return CrossFieldReport(
        field=raw,
        confidence=conf,
        mode=mode,
        anchors=anch,
        diagnostics=tuple(diagnostics),
        smoothed=bool(smoothed),
    )


# ---------------------------------------------------------------------------
# Candidate scoring and ranking
# ---------------------------------------------------------------------------


def score_body(
    state: QuadMeshState | View,
    report: CrossFieldReport,
    body: Sequence[int],
) -> float:
    """Mean alignment of a quad body's boundary edges with the local field.

    ``body`` is a canonical convex 4-walk (``make_quad`` output).  Each of the
    four boundary edges contributes the cosine of the angle between its
    fourfold representative and the field at its two endpoints (averaged), and
    the body score is the mean over the four edges, in ``[-1, 1]``.  Higher is
    better.  Boundary edges that are orthogonal pairs share the same fourfold
    representative, which is exactly the intended cross alignment signal.
    """
    b = tuple(int(x) for x in body)
    if len(b) != 4 or len(set(b)) != 4:
        raise GuidanceRejected(f"score body {b!r} must be a 4-node body")
    f = report.field
    positions = state.nodes
    terms: list[float] = []
    for i in range(4):
        a = b[i]
        c = b[(i + 1) % 4]
        rep = _edge_rep_4(positions, a, c)
        fa = f.get(a)
        fc = f.get(c)
        if fa is None and fc is None:
            continue
        dot = 0.0
        cnt = 0
        if fa is not None:
            dot += fa[0] * rep[0] + fa[1] * rep[1]
            cnt += 1
        if fc is not None:
            dot += fc[0] * rep[0] + fc[1] * rep[1]
            cnt += 1
        if cnt:
            terms.append(dot / cnt)
    if not terms:
        return 0.0
    return sum(terms) / len(terms)


def rank_bodies(
    state: QuadMeshState | View,
    report: CrossFieldReport,
    bodies: Sequence[Sequence[int]],
) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Deterministic (descending score, canonical body) ordering.

    Ties break by the lexicographic order of the body's node ids, so a shuffled
    candidate list always yields the same winner for an unchanged field.
    """
    scored = [(score_body(state, report, b), tuple(int(x) for x in b)) for b in bodies]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


# ---------------------------------------------------------------------------
# Guided front step (opt-in; unguided front_step is untouched)
# ---------------------------------------------------------------------------


def front_step_guided(
    state: QuadMeshState,
    edge: Sequence[int],
    *,
    report: CrossFieldReport,
    min_confidence: float = 0.0,
    options: QuadMeshingOptions | Mapping[str, Any] | None = None,
) -> tuple[int, tuple[int, int, int, int]]:
    """Advance the front at ``edge`` ranking admissible candidates by cross-field.

    Geometry / admissibility / protected / commit semantics are identical to
    :func:`anymesher.quad.front.front_step`; the only difference is the ordering
    of *admissible* candidates (descending :func:`score_body`), and the winner
    is committed.  Rejections (typed, state left untouched):

    * :class:`FrontNoCandidate` — edge not on front / no source / no partner /
      every candidate rejected (same rules as unguided).
    * :class:`LowConfidenceError` — the source's front-edge endpoint confidence
      (or any kept candidate endpoint) falls below ``min_confidence``.
    """
    if options is not None and not isinstance(options, (QuadMeshingOptions, Mapping)):
        raise MeshError("options must be None, QuadMeshingOptions or a mapping")

    min_conf = _validate_real(min_confidence, "min_confidence", lo=0.0, hi=1.0)

    fe = edge_key(int(edge[0]), int(edge[1]))
    if not state.is_front_edge(fe):
        raise FrontNoCandidate(f"edge {tuple(fe)} is not on the active front")
    try:
        source = find_source_cell(state, fe)
    except FrontRejected as exc:
        raise FrontNoCandidate(str(exc)) from exc

    # Confidence gate on the front edge endpoints.
    for n in (fe[0], fe[1]):
        c = report.confidence.get(n)
        if c is not None and c < min_conf:
            raise LowConfidenceError(
                f"confidence at node {n} is {c:.6f} below min_confidence {min_conf}"
            )

    partners = candidate_partners(state, source)
    if not partners:
        raise FrontNoCandidate(
            f"front edge {tuple(fe)}: source {source} has no T3 partner across a non-front edge"
        )

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    last_reject: str | None = None
    for partner in partners:
        try:
            body = classify(state, fe, source, partner)
        except FrontRejected as exc:
            last_reject = str(exc)
            continue
        # Confidence gate on candidate endpoints.
        ok = True
        for n in body:
            c = report.confidence.get(n)
            if c is not None and c < min_conf:
                last_reject = f"candidate endpoint {n} below min_confidence"
                ok = False
                break
        if ok:
            candidates.append((partner, body))

    if not candidates:
        raise FrontNoCandidate(
            f"front edge {tuple(fe)}: every candidate rejected ({last_reject})"
        )

    # Rank admissible candidates; commit the highest-scored, ties deterministic.
    best_body = rank_bodies(state, report, [body for _, body in candidates])[0][1]
    best_partner = next(partner for partner, body in candidates if body == best_body)

    _s_body = tuple(int(x) for x in state.cell(source))
    _p_body = tuple(int(x) for x in state.cell(best_partner))
    touched = (
        set(body_edges(_s_body)) | set(body_edges(_p_body)) | set(body_edges(best_body))
    )

    new_id = -1
    with state.transaction() as tx:
        tx.remove_cell(source)
        tx.remove_cell(best_partner)
        new_id = tx.allocate_cell(best_body, "Q4")

        for k in sorted(touched):
            is_front_now = sum(
                1 for cid in tx.view.edge_cells(k)
                if tx.view.cell_kind(cid) == "T3"
            ) == 1
            was_front = state.is_front_edge(k)
            if is_front_now and not was_front:
                tx.add_front_edge(k[0], k[1])
            elif (not is_front_now) and was_front:
                tx.remove_front_edge(k[0], k[1])

        tx.commit()  # atomic: prevalidates (incl. front residency) then applies

    return new_id, tuple(int(x) for x in best_body)
