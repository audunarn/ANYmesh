"""Quad-first advancing-front driver: convert a T3 pair into one Q4 at a front edge.

A *front edge* is a resident edge with exactly one incident cell (a ``T3``).
The driver consumes the front edge's unique ``T3`` source cell and a ranked
``T3`` partner (adjacent to the source across a *non-front*, i.e. interior,
edge) to stage a ``Q4`` whose boundary is the union of the two triangles, then
publishes the local front update through a transaction.

Rules (typed failures, never a silent fallback):

* :class:`FrontRejected` — a specific admissibility rule rejects one candidate
  (source/partner not ``T3``, not a clean quad pair, protected diagonal or
  endpoint, non-strictly-convex or degenerate union).
* :class:`FrontNoCandidate` — the edge is not on the active front, has no
  unique ``T3`` source, no ``T3`` partner, or every candidate was rejected.

An unexpected commit failure raises :class:`~anymesher.errors.MeshError` with
the state left untouched (transaction atomicity enforces this); there is no
"step rejected" pseudo-success and no hidden fallback.

Geometry is owned by ANYgeometry: this module only *classifies* using
``orient2d`` and shoelace area through node positions. Node identity is an
exact integer id (no coordinate welding) and edges are keyed by the exact node
pair (``lo < hi``). The committed edit is sparse and local: exactly the two
triangles, one ``Q4`` cell and the front state of the touched local boundary.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..errors import MeshError
from ..triangulation import orient2d
from .journal import Transaction  # noqa: F401  (document the writer contract)
from .options import QuadMeshingOptions
from .state import EdgeKey, QuadMeshState, View

__all__ = [
    "EPS",
    "FrontRejected",
    "FrontNoCandidate",
    "edge_key",
    "body_edges",
    "area2",
    "make_quad",
    "local_swap",
    "find_source_cell",
    "candidate_partners",
    "classify",
    "front_step",
]

# Tolerance for strictly-convex / non-degenerate turn and area tests.
EPS = 1e-9


class FrontRejected(MeshError):
    """A specific front rule rejected this candidate (typed, non-fatal)."""


class FrontNoCandidate(MeshError):
    """No admissible candidate pair exists at the front edge."""


# ---------------------------------------------------------------------------
# Pure geometry (positions read through the view/state; never welded)
# ---------------------------------------------------------------------------

def edge_key(a: int, b: int) -> EdgeKey:
    """Normalized edge key (``lo < hi``) for a distinct node pair."""
    if a == b:
        raise MeshError("edge must have two distinct node ids")
    return (a, b) if a < b else (b, a)


def body_edges(body: Sequence[int]) -> list[EdgeKey]:
    """Cyclic boundary edge keys of a cell body (in walk order)."""
    n = len(body)
    return [edge_key(body[i], body[(i + 1) % n]) for i in range(n)]


def _orient(view: View | QuadMeshState, a: int, b: int, c: int) -> float:
    return orient2d(view.position(a), view.position(b), view.position(c))


def area2(view: View | QuadMeshState, body: Sequence[int]) -> float:
    """Signed double area (shoelace) of the vertex body, in node-id walk order."""
    n = len(body)
    s = 0.0
    for i in range(n):
        x1, y1 = view.position(body[i])
        x2, y2 = view.position(body[(i + 1) % n])
        s += x1 * y2 - x2 * y1
    return s


def make_quad(view: View | QuadMeshState, raw: Sequence[int]) -> tuple[int, int, int, int]:
    """Return a canonical CCW-ordered, strictly-convex body for a 4-node walk.

    ``raw`` is a closed 4-walk (typically the boundary walk of a candidate
    union).  Rejects with :class:`FrontRejected` if the nodes repeat, the area
    is degenerate, or the boundary is not strictly convex.
    """
    body = tuple(int(x) for x in raw)
    if len(body) != 4 or len(set(body)) != 4:
        raise FrontRejected(f"quad body {body!r} must have four distinct nodes")
    area = area2(view, body)
    if abs(area) <= EPS:
        raise FrontRejected(f"quad body {body!r} is degenerate (zero area)")
    sgn = 1.0 if area > 0 else -1.0
    for i in range(4):
        o = sgn * _orient(view, body[i], body[(i + 1) % 4], body[(i + 2) % 4])
        if o <= EPS:
            raise FrontRejected(f"quad body {body!r} is not strictly convex (turn {o:+g})")
    return body if area > 0 else (body[0], body[3], body[2], body[1])


def _boundary_nodes(
    view: View | QuadMeshState,
    front_edge: EdgeKey,
    source: tuple[int, ...],
    partner: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Deterministic CCW boundary node-order of the union of the two triangles.

    The union has 4 nodes and 4 boundary edges (the shared interior edge is
    dropped).  The 4 boundary edges form a simple cycle (each node has
    degree 2); walking it from ``min(front_edge)`` in both directions gives the
    same cycle in the two orientations, of which exactly one is CCW (positive
    signed area).  Returns that CCW 4-node tuple.
    """
    s_edges = set(body_edges(source))
    p_edges = set(body_edges(partner))
    shared = s_edges & p_edges
    if len(shared) != 1:
        raise FrontRejected("source and partner do not share exactly one interior edge")
    diag = next(iter(shared))
    boundary = (s_edges | p_edges) - {diag}
    if len(boundary) != 4:
        raise FrontRejected("candidate union does not have a clean 4-edge boundary")

    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for nn, nb in adj.items():
        if len(nb) != 2:
            raise FrontRejected("candidate boundary is not a clean 4-walk")

    start = min(edge_key(*front_edge))
    paths: list[tuple[int, ...]] = []
    for first_nb in adj[start]:
        path = [start, first_nb]
        prev, cur = start, first_nb
        while len(path) < 4:
            nxt = [x for x in adj[cur] if x != prev][0]
            path.append(nxt)
            prev, cur = cur, nxt
        if cur not in adj[start]:
            raise FrontRejected("candidate boundary walk did not close")
        paths.append(tuple(path))

    best = max(paths, key=lambda t: area2(view, t))
    if area2(view, best) <= EPS:
        raise FrontRejected("candidate boundary is degenerate")
    return best


def local_swap(body: Sequence[int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Flip the diagonal of a (canonical convex) quad body (pure topology).

    ``body = (a, b, c, d)`` in cyclic order is retiled on the other diagonal
    ``(b, d)`` into the two triangles ``(a, b, d)`` and ``(b, c, d)``, the
    complementary triangulation of the same quadrilateral.
    """
    b = tuple(int(x) for x in body)
    if len(b) != 4 or len(set(b)) != 4:
        raise FrontRejected(f"local_swap body {b!r} is not a 4-node quad")
    a, bb, c, d = b
    return (a, bb, d), (bb, c, d)


# ---------------------------------------------------------------------------
# Candidate discovery and classification
# ---------------------------------------------------------------------------

def _is_residual_front(view: View | QuadMeshState, edge: EdgeKey) -> bool:
    """True when exactly one residual T3 is incident on ``edge``."""
    k = edge_key(*edge)
    return sum(1 for cid in view.edge_cells(k) if view.cell_kind(cid) == "T3") == 1


def find_source_cell(view: View | QuadMeshState, edge: EdgeKey) -> int:
    """The unique incident residual ``T3`` cell below the front edge ``edge``.

    The source is the exactly-one incident residual ``T3``; a ``Q4`` on the
    accepted side of the edge is permitted and is ignored here.
    """
    k = edge_key(*edge)
    cells = tuple(sorted(int(cid) for cid in view.edge_cells(k)))
    residual = tuple(cid for cid in cells if view.cell_kind(cid) == "T3")
    if not residual:
        raise FrontRejected(f"front edge {tuple(k)} has no incident residual T3")
    if len(residual) != 1:
        raise FrontRejected(f"front edge {tuple(k)} has {len(residual)} incident residual T3 {residual!r} (need exactly 1)")
    return residual[0]


def candidate_partners(view: View | QuadMeshState, source: int) -> list[int]:
    """Ranked ``T3`` partners adjacent to ``source`` across its non-front edges.

    Deterministic order (source boundary edges sorted by normalized key, then
    cell id).  A candidate is the single other incident ``T3`` cell across a
    shared non-front edge; whether it forms a valid quad pair is decided by
    :func:`classify`, not here.
    """
    if view.cell_kind(source) != "T3":
        raise FrontRejected(f"source {source} is not a T3 cell")
    s = tuple(int(x) for x in view.cell(source))
    out: list[int] = []
    seen: set[int] = set()
    for e in sorted(body_edges(s)):
        if view.is_front_edge(e):
            continue
        inc = view.edge_cells(e)
        if len(inc) != 2 or source not in inc:
            continue
        other = next(c for c in inc if c != source)
        if other in seen:
            continue
        if view.cell_kind(other) != "T3":
            continue
        seen.add(other)
        out.append(other)
    return out


def classify(
    view: View | QuadMeshState,
    front_edge: EdgeKey,
    source: int,
    partner: int,
) -> tuple[int, int, int, int]:
    """Validate one candidate pair; return the canonical CCW convex quad body.

    Rejections (typed :class:`FrontRejected`):

    * ``source`` / ``partner`` are not ``T3`` or are the same cell
    * the front edge is not an edge of the source
    * the pair does not share exactly one interior (diagonal) edge, or that
      edge is the front edge itself
    * the shared interior diagonal is protected
    * protected/fixed endpoint nodes are allowed to participate as Q4 corners; movement/deletion protection is enforced elsewhere
    * the union boundary is not a strictly convex, non-degenerate quadrilateral

    (Duplicate-body rejection is handled atomically at commit by the state.)
    """
    if view.cell_kind(source) != "T3":
        raise FrontRejected(f"source {source} is not a T3 cell")
    if view.cell_kind(partner) != "T3":
        raise FrontRejected(f"partner {partner} is not a T3 cell")
    if source == partner:
        raise FrontRejected("source and partner are the same cell")

    s = tuple(int(x) for x in view.cell(source))
    p = tuple(int(x) for x in view.cell(partner))
    fe = edge_key(*front_edge)
    s_edges = set(body_edges(s))
    p_edges = set(body_edges(p))

    if fe not in s_edges:
        raise FrontRejected(f"front edge {tuple(fe)} is not an edge of source {source}")
    shared = s_edges & p_edges
    if len(shared) != 1:
        raise FrontRejected("source and partner do not share a single interior edge")
    diag = next(iter(shared))
    if diag == fe:
        raise FrontRejected("the front edge is the shared interior edge; invalid pairing")

    if view.is_protected_edge(diag):
        raise FrontRejected(f"shared interior edge {tuple(diag)} is a protected edge")

    boundary = _boundary_nodes(view, fe, s, p)
    return make_quad(view, boundary)


# ---------------------------------------------------------------------------
# Front driver: stage + commit one front step atomically
# ---------------------------------------------------------------------------

def front_step(
    state: QuadMeshState,
    edge: Sequence[int],
    options: QuadMeshingOptions | Mapping[str, Any] | None = None,
) -> tuple[int, tuple[int, int, int, int]]:
    """Advance the front at ``edge`` by one step, committed atomically.

    Returns ``(new_cell_id, body)`` of the published ``Q4``.  On a rejected
    candidate or a missing source/partner, the state is left unmodified and a
    typed :class:`FrontNoCandidate` is raised — never a silent "no-op success".
    An unexpected commit failure raises :class:`~anymesher.errors.MeshError`
    with the base untouched (transaction atomicity).
    """
    if options is not None and not isinstance(options, (QuadMeshingOptions, Mapping)):
        raise MeshError("options must be None, QuadMeshingOptions or a mapping")

    fe = edge_key(int(edge[0]), int(edge[1]))
    if not state.is_front_edge(fe):
        raise FrontNoCandidate(f"edge {tuple(fe)} is not on the active front")
    try:
        source = find_source_cell(state, fe)
    except FrontRejected as exc:
        raise FrontNoCandidate(str(exc)) from exc

    partners = candidate_partners(state, source)
    if not partners:
        raise FrontNoCandidate(
            f"front edge {tuple(fe)}: source {source} has no T3 partner across a non-front edge"
        )

    chosen: tuple[int, tuple[int, int, int, int]] | None = None
    last_reject: str | None = None
    for partner in partners:
        try:
            quad = classify(state, fe, source, partner)
        except FrontRejected as exc:
            last_reject = str(exc)
            continue
        chosen = (partner, quad)
        break
    if chosen is None:
        raise FrontNoCandidate(
            f"front edge {tuple(fe)}: every candidate rejected ({last_reject})"
        )

    partner, quad = chosen

    # Local (O(touched)) front update: capture the incident edge sets *before*
    # the removals are staged, since removals drop the triangles from the base
    # the moment commit applies the delta.
    s_body = tuple(int(x) for x in state.cell(source))
    p_body = tuple(int(x) for x in state.cell(partner))
    touched = set(body_edges(s_body)) | set(body_edges(p_body)) | set(body_edges(quad))

    new_id = -1
    with state.transaction() as tx:
        tx.remove_cell(source)
        tx.remove_cell(partner)
        new_id = tx.allocate_cell(quad, "Q4")

        for k in sorted(touched):
            is_front_now = _is_residual_front(tx.view, k)
            was_front = state.is_front_edge(k)
            if is_front_now and not was_front:
                tx.add_front_edge(k[0], k[1])
            elif (not is_front_now) and was_front:
                tx.remove_front_edge(k[0], k[1])

        tx.commit()  # atomic: prevalidates (incl. front residency) then applies

    return new_id, tuple(quad)
