"""Resident local mixed T3/Q4 mesh state for the quad-first advancing front.

Adapted from the Q-Morph/Owen-Cannan (1998) advancing-front algorithm
(referenced in ``third_party/quad/ATTRIBUTION.md``).

Architecture (final for Q1, per programme decision):

* :class:`QuadMeshState` is the **resident base** mesh: node positions, mixed
  T3/Q4 cells, edge-to-cell incidence, node-to-edge adjacency, the front
  edge set plus its per-endpoint side bits, protected features, and a
  monotonically increasing *generation* counter.  The driver owns it and the
  transaction reads it.
* A :class:`Delta` records a **sparse local edit** (nodes/cells/front/bits/
  protected) relative to that base, plus lazily-derived touched indices.
* :class:`View` is a thin read-only candidate proxy: every read combines
  ``base`` with the delta and only touches the touched local keys, so no
  full-mesh copy or per-edit export happens (O(local-edit), not O(full)).
* Rollback is **discard-only** (throw the delta away).  Commit
  pre-validates the full delta against the base (generation, protected
  features, incidence, duplicate/degenerate cells, cancellation checkpoint)
  and then applies it in one short deterministic non-cancellable section via
  :meth:`QuadMeshState._apply_delta`, which also bumps the generation.

Geometry is never welded: node identity is an exact integer id owned by
ANYgeometry and edges are keyed by the exact node pair (``lo < hi``), never
by coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..errors import MeshError

__all__ = [
    "QuadMeshState",
    "Delta",
    "View",
    "StaleHandleError",
    "CancellationRequested",
    "EdgeKey",
]


class StaleHandleError(MeshError):
    """Raised when a node/cell/edge id is not resident in the state under query."""


class CancellationRequested(MeshError):
    """Raised at a cancellation checkpoint when the caller requested stop."""


EdgeKey = tuple[int, int]


def _edge_key(a: int, b: int) -> EdgeKey:
    if a == b:
        raise MeshError("edge must have two distinct node ids")
    return (a, b) if a < b else (b, a)


def _int_or_err(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeshError(f"{name} must be an integer")
    return value


def _check_position(pos: Sequence[float]) -> tuple[float, float]:
    if len(pos) != 2:
        raise MeshError("each node needs exactly 2 coordinates")
    try:
        x = float(pos[0])
        y = float(pos[1])
    except (TypeError, ValueError) as exc:
        raise MeshError("node positions must be numeric") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise MeshError("node positions must be finite")
    return (x, y)


def _cell_edges(body: tuple[int, int, ...]) -> list[EdgeKey]:
    return [_edge_key(body[i], body[(i + 1) % len(body)]) for i in range(len(body))]


# ---------------------------------------------------------------------------
# Sparse local delta
# ---------------------------------------------------------------------------

@dataclass
class Delta:
    """A sparse local edit relative to a resident base state.

    Candidate incidence is always computed on demand via :meth:`touched_map`,
    which maps an edge key to the net cell-incidence change (`{cell_id: +1 or
    -1}`) from the CURRENT cell add/remove sets plus the front/bit/protected
    edge sets.  Nothing is cached across mutations, so staging further edits
    after a view was taken never leaves stale candidate incidence behind.
    """

    add_nodes: dict[int, tuple[float, float]] = field(default_factory=dict)
    remove_nodes: set[int] = field(default_factory=set)
    move_nodes: dict[int, tuple[float, float]] = field(default_factory=dict)
    add_cells: dict[int, tuple[int, int, ...]] = field(default_factory=dict)
    add_cell_kinds: dict[int, str] = field(default_factory=dict)
    remove_cells: set[int] = field(default_factory=set)
    add_front: set[EdgeKey] = field(default_factory=set)
    remove_front: set[EdgeKey] = field(default_factory=set)
    add_bits: set[tuple[EdgeKey, int]] = field(default_factory=set)
    remove_bits: set[tuple[EdgeKey, int]] = field(default_factory=set)
    add_prot_nodes: set[int] = field(default_factory=set)
    remove_prot_nodes: set[int] = field(default_factory=set)
    add_prot_edges: set[EdgeKey] = field(default_factory=set)
    remove_prot_edges: set[EdgeKey] = field(default_factory=set)

    def is_empty(self) -> bool:
        self._cancel_pairs()
        return not (
            self.add_nodes
            or self.remove_nodes
            or self.move_nodes
            or self.add_cells
            or self.remove_cells
            or self.add_front
            or self.remove_front
            or self.add_bits
            or self.remove_bits
            or self.add_prot_nodes
            or self.remove_prot_nodes
            or self.add_prot_edges
            or self.remove_prot_edges
        )

    def _cancel_pairs(self) -> None:
        """Drop entries staged on both sides of an add/remove pair.

        Staging ``protect_edge(e)`` then ``release_edge(e)`` (or front/bit/
        node/cell pairs) is a net no-op; both sides are removed so commit and
        candidate views see the true net edit instead of applying the add over
        the remove (order of the two sets is meaningless).
        """
        for add, rem in (
            (self.add_front, self.remove_front),
            (self.add_bits, self.remove_bits),
            (self.add_prot_nodes, self.remove_prot_nodes),
            (self.add_prot_edges, self.remove_prot_edges),
        ):
            both = add & rem
            if both:
                add -= both
                rem -= both
        node_both = set(self.add_nodes) & self.remove_nodes
        for n in node_both:
            del self.add_nodes[n]
            self.remove_nodes.discard(n)
        cell_both = set(self.add_cells) & self.remove_cells
        for cid in cell_both:
            self.add_cells.pop(cid, None)
            self.add_cell_kinds.pop(cid, None)
            self.remove_cells.discard(cid)
        # Reject moved-node conflicts with add_nodes or remove_nodes
        move_conflict = set(self.move_nodes) & (set(self.add_nodes) | self.remove_nodes)
        if move_conflict:
            raise MeshError(f"nodes both moved and added/removed: {sorted(move_conflict)!r}")

    def touched_map(self, base: "QuadMeshState") -> dict[EdgeKey, dict[int, int]]:
        """Pure O(local) net cell-incidence map for the CURRENT sparse contents.

        Maps each candidate-touched edge key to ``{cell_id: +1/-1}`` added to
        (resp. removed from) the resident incidence.  Always derived from live
        contents so reads after further staging stay correct.
        """
        touched: dict[EdgeKey, dict[int, int]] = {}

        def note_edge(k: EdgeKey, cid: int, sign: int) -> None:
            cellmap = touched.setdefault(k, {})
            cellmap[cid] = cellmap.get(cid, 0) + sign

        for cid, body in self.add_cells.items():
            for k in _cell_edges(body):
                note_edge(k, cid, +1)
        for cid in self.remove_cells:
            if cid not in base.cells:
                continue
            body = base.cells[cid]
            for k in _cell_edges(body):
                note_edge(k, cid, -1)

        for edge_set in (
            self.add_front,
            self.remove_front,
            self.add_prot_edges,
            self.remove_prot_edges,
        ):
            for k in edge_set:
                touched.setdefault(k, {})
        for k, _n in self.add_bits:
            touched.setdefault(k, {})
        for k, _n in self.remove_bits:
            touched.setdefault(k, {})

        return touched


# ---------------------------------------------------------------------------
# Resident base state
# ---------------------------------------------------------------------------

class QuadMeshState:
    """Resident mixed T3/Q4 local mesh (base for transactions)."""

    def __init__(
        self,
        nodes: Mapping[int, Sequence[float]],
        cells: Mapping[int, Sequence[int]],
        cell_kinds: Mapping[int, str] | None = None,
        initial_front: Iterable[Sequence[int]] | None = None,
        protected_nodes: Iterable[int] = (),
        protected_edges: Iterable[Sequence[int]] = (),
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if cell_kinds is None:
            cell_kinds = {}

        pos: dict[int, tuple[float, float]] = {}
        for n, p in nodes.items():
            pos[_int_or_err(n, "node")] = _check_position(p)

        cells_map: dict[int, tuple[int, int, ...]] = {}
        kinds: dict[int, str] = {}
        for cid_raw, body in cells.items():
            cid = _int_or_err(cid_raw, "cell")
            kind = cell_kinds.get(cid, "")
            body = tuple(_int_or_err(x, "cell node") for x in body)
            if kind == "":
                kind = "Q4" if len(body) == 4 else "T3"
            if kind not in ("T3", "Q4"):
                raise MeshError(f"cell {cid} has unknown kind {kind!r}")
            if len(body) != (3 if kind == "T3" else 4):
                raise MeshError(f"cell {cid} kind {kind} needs {3 if kind == 'T3' else 4} nodes")
            if len(set(body)) != len(body):
                raise MeshError(f"cell {cid} repeats a node")
            for n in body:
                if n not in pos:
                    raise MeshError(f"cell {cid} references unknown node {n}")
            cells_map[cid] = body
            kinds[cid] = kind

        self._pos = pos
        self._cells = cells_map
        self._kind = kinds
        # Maintained with every commit so drivers can ask "any residual T3?"
        # in O(1) instead of scanning all cells per front step.
        self._kind_count: dict[str, int] = {"T3": 0, "Q4": 0}
        for kind in kinds.values():
            self._kind_count[kind] += 1
        self._edge_to_cells: dict[EdgeKey, set[int]] = {}
        self._node_edges: dict[int, set[EdgeKey]] = {n: set() for n in pos}
        self._body_to_cell: dict[tuple[int, int, ...], int] = {}
        for cid, body in cells_map.items():
            self._body_to_cell[tuple(sorted(body))] = cid
            for i in range(len(body)):
                k = _edge_key(body[i], body[(i + 1) % len(body)])
                self._edge_to_cells.setdefault(k, set()).add(cid)
                self._node_edges.setdefault(k[0], set()).add(k)
                self._node_edges.setdefault(k[1], set()).add(k)

        self._front: set[EdgeKey] = set()
        if initial_front is not None:
            for e in initial_front:
                k = _edge_key(_int_or_err(e[0], "front node"), _int_or_err(e[1], "front node"))
                if k not in self._edge_to_cells:
                    raise MeshError(f"front edge {k} is not resident")
                self._front.add(k)
        self._front_bits: set[tuple[EdgeKey, int]] = set()
        self._prot_nodes: set[int] = set(
            _int_or_err(n, "protected node") for n in protected_nodes
        )
        self._prot_edges: set[EdgeKey] = set()
        for e in protected_edges:
            k = _edge_key(_int_or_err(e[0], "protected node"), _int_or_err(e[1], "protected node"))
            if k not in self._edge_to_cells:
                raise MeshError(f"protected edge {k} is not resident")
            self._prot_edges.add(k)

        self._is_cancelled = is_cancelled
        self._generation = 0
        self._next_node_id = (max(pos) + 1) if pos else 0
        self._next_cell_id = (max(cells_map) + 1) if cells_map else 0

    # ------------------------------------------------------------------
    # Read-only access (base)
    # ------------------------------------------------------------------

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def next_node_id(self) -> int:
        """Next node id a *committed* addition will receive."""
        return self._next_node_id

    @property
    def next_cell_id(self) -> int:
        """Next cell id a *committed* addition will receive."""
        return self._next_cell_id

    @property
    def node_ids(self) -> frozenset[int]:
        return frozenset(self._pos)

    @property
    def nodes(self) -> Mapping[int, tuple[float, float]]:
        return self._pos

    @property
    def cells(self) -> Mapping[int, tuple[int, int, ...]]:
        return self._cells

    @property
    def cell_kinds(self) -> Mapping[int, str]:
        return self._kind

    @property
    def front(self) -> frozenset[EdgeKey]:
        return frozenset(self._front)

    @property
    def front_bits(self) -> frozenset[tuple[EdgeKey, int]]:
        return frozenset(self._front_bits)

    @property
    def protected_nodes(self) -> frozenset[int]:
        return frozenset(self._prot_nodes)

    @property
    def protected_edges(self) -> frozenset[EdgeKey]:
        return frozenset(self._prot_edges)

    @property
    def edges(self) -> frozenset[EdgeKey]:
        return frozenset(self._edge_to_cells)

    def position(self, n: Any) -> tuple[float, float]:
        nid = _int_or_err(n, "node")
        if nid not in self._pos:
            raise StaleHandleError(f"node {nid} is not a live node in this state")
        return self._pos[nid]

    def cell(self, cid: Any) -> tuple[int, int, ...]:
        c = _int_or_err(cid, "cell")
        if c not in self._cells:
            raise StaleHandleError(f"cell {c} does not exist (stale handle)")
        return self._cells[c]

    def cell_kind(self, cid: Any) -> str:
        return self._kind[_int_or_err(cid, "cell")]

    def count_kind(self, kind: str) -> int:
        """Number of resident cells of ``kind`` (``"T3"`` or ``"Q4"``), O(1)."""
        if kind not in self._kind_count:
            raise MeshError(f"unknown cell kind {kind!r}")
        return self._kind_count[kind]

    def edge_cells(self, key: Any) -> tuple[int, ...]:
        k = _edge_key(_int_or_err(key[0], "edge[0]"), _int_or_err(key[1], "edge[1]"))
        return tuple(sorted(self._edge_to_cells.get(k, ())))

    def neighbors_at(self, n: Any) -> tuple[EdgeKey, ...]:
        nid = _int_or_err(n, "node")
        if nid not in self._pos:
            raise StaleHandleError(f"node {nid} is not a live node in this state")
        return tuple(sorted(self._node_edges.get(nid, ())))

    def front_neighbors_at(self, n: Any, *, exclude: EdgeKey | None = None) -> tuple[EdgeKey, ...]:
        nid = _int_or_err(n, "node")
        if nid not in self._pos:
            raise StaleHandleError(f"node {nid} is not a live node in this state")
        return tuple(sorted(k for k in self._front if nid in k and k != exclude))

    def cells_at(self, n: Any) -> tuple[int, ...]:
        nid = _int_or_err(n, "node")
        if nid not in self._pos:
            raise StaleHandleError(f"node {nid} is not a live node in this state")
        return tuple(sorted(c for c, b in self._cells.items() if nid in b))

    def cells_at_edge(self, key: Any) -> tuple[int, ...]:
        return self.edge_cells(key)

    def other_node_of(self, key: Any, n: Any) -> int:
        nid = _int_or_err(n, "node")
        k = _edge_key(_int_or_err(key[0], "edge[0]"), _int_or_err(key[1], "edge[1]"))
        if nid not in k:
            raise MeshError(f"node {nid} is not on edge {k}")
        return k[0] if k[1] == nid else k[1]

    def is_front_edge(self, key: Any) -> bool:
        try:
            k = _edge_key(_int_or_err(key[0], "edge[0]"), _int_or_err(key[1], "edge[1]"))
        except (MeshError, StaleHandleError):
            return False
        return k in self._front

    def bit_set(self, edge: Sequence[int], n: Any) -> bool:
        k = _edge_key(_int_or_err(edge[0], "edge[0]"), _int_or_err(edge[1], "edge[1]"))
        return (k, _int_or_err(n, "bit node")) in self._front_bits

    def is_protected_node(self, n: Any) -> bool:
        try:
            return _int_or_err(n, "protected node") in self._prot_nodes
        except (MeshError, StaleHandleError):
            return False

    def is_protected_edge(self, key: Any) -> bool:
        try:
            k = _edge_key(_int_or_err(key[0], "edge[0]"), _int_or_err(key[1], "edge[1]"))
        except (MeshError, StaleHandleError):
            return False
        return k in self._prot_edges

    def is_cancelled(self) -> bool:
        if self._is_cancelled is None:
            return False
        return bool(self._is_cancelled())

    def checkpoint(self) -> None:
        if self.is_cancelled():
            raise CancellationRequested("cancellation requested at checkpoint")

    # ------------------------------------------------------------------
    # Candidate view
    # ------------------------------------------------------------------

    def combine(self, delta: Delta) -> View:
        return View(self, delta)

    # ------------------------------------------------------------------
    # Transaction factory
    # ------------------------------------------------------------------

    def transaction(self) -> "Transaction":
        """Return a new :class:`~.journal.Transaction` writing this state.

        The transaction captures the current generation; committing it atomically
        applies the staged delta and bumps :attr:`generation`.
        """
        from .journal import Transaction
        return Transaction(self)

    # ------------------------------------------------------------------
    # Digests / introspection (tests)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return self._dump()

    def _dump(self) -> dict[str, Any]:
        return {
            "generation": self._generation,
            "nodes": {n: list(p) for n, p in sorted(self._pos.items())},
            "cells": {c: list(b) for c, b in sorted(self._cells.items())},
            "cell_kinds": {c: k for c, k in sorted(self._kind.items())},
            "edge_cells": {
                f"{a},{b}": sorted(v) for (a, b), v in sorted(self._edge_to_cells.items())
            },
            "front": [[a, b] for a, b in sorted(self._front)],
            "front_bits": [[a, b, n] for (a, b), n in sorted(self._front_bits)],
            "protected_nodes": sorted(self._prot_nodes),
            "protected_edges": [[a, b] for a, b in sorted(self._prot_edges)],
        }

    def digest(self) -> str:
        payload = json.dumps(
            self._dump(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Atomic delta application (only a transaction's commit reaches this)
    # ------------------------------------------------------------------

    def _apply_delta(self, delta: Delta) -> None:
        """Prevalidate the full delta, then apply it as a local in-place patch.

        *Pass 1 (validation only, no mutation):* using the resident edge,
        node, front, protected and normalized-body indices plus the sparse
        touched data, check stale removes, added-cell wellformedness, duplicate
        cell bodies (against ``self._body_to_cell`` with removed bodies freed),
        front/bit/protection residency, and the cancellation checkpoint.

        *Pass 2 (mutating, cannot fail):* patch only the touched nodes,
        cells, body index, edge/incidence sets, node adjacency, front, bits
        and protected features, then bump :attr:`generation`.  No whole-mesh
        cells/edges/node-adjacency reconstruction happens per commit.
        """
        # ---------------- Pass 1: validate (pure reads) ----------------
        touched = delta.touched_map(self)

        removed_bodies = {
            cid: self._cells[cid] for cid in delta.remove_cells if cid in self._cells
        }
        for cid in delta.remove_cells:
            if cid not in self._cells:
                raise StaleHandleError(f"cell {cid} does not exist (stale handle)")

        checked_add_nodes = {n: _check_position(p) for n, p in delta.add_nodes.items()}

        def final_node_live(n: int) -> bool:
            if n in delta.remove_nodes:
                return False
            if n in delta.add_nodes:
                return True
            return n in self._pos

        for n in delta.remove_nodes:
            _int_or_err(n, "node")
        conflict = set(delta.add_nodes) & set(delta.remove_nodes)
        if conflict:
            raise MeshError(f"nodes both added and removed: {sorted(conflict)!r}")

        checked_move_nodes: dict[int, tuple[float, float]] = {}
        for n, p in delta.move_nodes.items():
            n = _int_or_err(n, "moved node")
            if n in delta.add_nodes or n in delta.remove_nodes:
                raise MeshError(f"node {n} cannot be both moved and added/removed")
            if n not in self._pos:
                raise StaleHandleError(f"node {n} is not a live node in this state")
            if n in self._prot_nodes:
                raise MeshError(f"node {n} is protected and cannot be moved")
            checked_move_nodes[n] = _check_position(p)

        freed_bodies = {tuple(sorted(body)) for body in removed_bodies.values()}

        def final_resident(k: EdgeKey) -> bool:
            inc = set(self._edge_to_cells.get(k, ()))
            net = touched.get(k)
            if net:
                for cid, sign in net.items():
                    if sign > 0:
                        inc.add(cid)
                    elif sign < 0:
                        inc.discard(cid)
            return bool(inc)

        added_norms: dict[tuple[int, int, ...], int] = {}
        for cid in sorted(delta.add_cells):
            body = delta.add_cells[cid]
            if len(body) not in (3, 4):
                raise MeshError(f"cell {cid} is not T3/Q4")
            if len(set(body)) != len(body):
                raise MeshError(f"cell {cid} repeats a node")
            for n in body:
                if not final_node_live(n):
                    raise MeshError(f"cell {cid} references unknown node {n}")
            kind = delta.add_cell_kinds.get(cid, "Q4" if len(body) == 4 else "T3")
            if kind not in ("T3", "Q4"):
                raise MeshError(f"cell {cid} has unknown kind {kind!r}")
            if len(body) != (3 if kind == "T3" else 4):
                raise MeshError(f"cell {cid} kind {kind} needs {3 if kind == 'T3' else 4} nodes")
            norm = tuple(sorted(body))
            if norm in added_norms:
                raise MeshError(
                    f"duplicate cell body shared by {added_norms[norm]} and {cid}"
                )
            if norm not in freed_bodies and norm in self._body_to_cell:
                raise MeshError(
                    f"duplicate cell body shared by {self._body_to_cell[norm]} and {cid}"
                )
            added_norms[norm] = cid

        # Front / bits / protected: validate only locally changed/touched keys.
        def final_front(k: EdgeKey) -> bool:
            if k in delta.add_front:
                return True
            if k in delta.remove_front:
                return False
            return k in self._front

        def final_bit(k: EdgeKey, n: int) -> bool:
            pair = (k, n)
            if pair in delta.add_bits:
                return True
            if pair in delta.remove_bits:
                return False
            return pair in self._front_bits

        front_check_edges = set(touched) | delta.add_front | delta.remove_front
        front_check_edges |= {k for k, _n in delta.add_bits}
        front_check_edges |= {k for k, _n in delta.remove_bits}
        for k in sorted(front_check_edges):
            is_front = final_front(k)
            if is_front and not final_resident(k):
                raise MeshError(f"front edge {k} is not resident in the candidate")
            for n in k:
                if final_bit(k, n) and not is_front:
                    raise MeshError(f"bit edge {k} is not a front edge")
        for k, n in sorted(delta.add_bits):
            if n not in k:
                raise MeshError(f"bit endpoint {n} is not on edge {k}")
            if not final_front(k):
                raise MeshError(f"bit edge {k} is not a front edge")

        for n in sorted(delta.add_prot_nodes):
            if not final_node_live(n):
                raise MeshError(f"protected node {n} is not live")
        for n in sorted(delta.remove_nodes):
            if n in self._prot_nodes and n not in delta.remove_prot_nodes:
                raise MeshError(f"protected node {n} cannot be removed")
        protected_check_edges = set(touched) | delta.add_prot_edges
        for k in sorted(protected_check_edges):
            if k in delta.remove_prot_edges:
                continue
            if (k in delta.add_prot_edges or k in self._prot_edges) and not final_resident(k):
                raise MeshError(f"protected edge {k} is not resident")

        self.checkpoint()  # cancellation gate before any mutation

        # ---------------- Pass 2: local in-place patch ----------------
        for n in delta.remove_nodes:
            self._pos.pop(n, None)
            self._node_edges.pop(n, None)
        for n, p in checked_add_nodes.items():
            self._pos[n] = p
            self._node_edges.setdefault(n, set())
        for n, p in checked_move_nodes.items():
            self._pos[n] = p

        for cid in sorted(delta.remove_cells):
            body = self._cells[cid]
            norm = tuple(sorted(body))
            if self._body_to_cell.get(norm) == cid:
                del self._body_to_cell[norm]
            for k in _cell_edges(body):
                inc = self._edge_to_cells.get(k)
                if inc is not None:
                    inc.discard(cid)
                    # Only prune the node-adjacency of an edge that lost its
                    # LAST incident cell; other cells may still cover it.
                    if not inc:
                        del self._edge_to_cells[k]
                        self._node_edges.get(k[0], set()).discard(k)
                        self._node_edges.get(k[1], set()).discard(k)
            self._cells.pop(cid, None)
            removed_kind = self._kind.pop(cid, None)
            if removed_kind is not None:
                self._kind_count[removed_kind] -= 1

        for cid in sorted(delta.add_cells):
            body = delta.add_cells[cid]
            self._body_to_cell[tuple(sorted(body))] = cid
            for i in range(len(body)):
                k = _edge_key(body[i], body[(i + 1) % len(body)])
                self._edge_to_cells.setdefault(k, set()).add(cid)
                self._node_edges.setdefault(k[0], set()).add(k)
                self._node_edges.setdefault(k[1], set()).add(k)
            self._cells[cid] = body
            added_kind = delta.add_cell_kinds.get(
                cid, "Q4" if len(body) == 4 else "T3"
            )
            self._kind[cid] = added_kind
            self._kind_count[added_kind] += 1

        self._front.difference_update(delta.remove_front)
        self._front.update(delta.add_front)
        self._front_bits.difference_update(delta.remove_bits)
        self._front_bits.update(delta.add_bits)
        self._prot_nodes.difference_update(delta.remove_prot_nodes)
        self._prot_nodes.update(delta.add_prot_nodes)
        self._prot_edges.difference_update(delta.remove_prot_edges)
        self._prot_edges.update(delta.add_prot_edges)
        if delta.add_nodes:
            self._next_node_id = max(self._next_node_id, max(delta.add_nodes) + 1)
        if delta.add_cells:
            self._next_cell_id = max(self._next_cell_id, max(delta.add_cells) + 1)
        self._generation += 1


# ---------------------------------------------------------------------------
# Candidate view: base + sparse delta (thin, O(local) reads)
# ---------------------------------------------------------------------------

class View:
    """Read-only candidate combining a resident base state and a sparse delta."""

    def __init__(self, base: QuadMeshState, delta: Delta) -> None:
        self._base = base
        # The view is a *live* candidate: incidence reads re-derive the touched
        # map from the CURRENT contents of this shared delta, so staging further
        # edits after a view was taken is always visible (and commit prevalidates
        # exactly what views show).
        self._delta = delta

    def _touched(self) -> dict[EdgeKey, dict[int, int]]:
        return self._delta.touched_map(self._base)

    @property
    def base(self) -> QuadMeshState:
        return self._base

    @property
    def delta(self) -> Delta:
        return self._delta

    # -- nodes -----------------------------------------------------------

    @property
    def node_ids(self) -> frozenset[int]:
        d = self._delta
        return frozenset((set(self._base.node_ids) - d.remove_nodes) | set(d.add_nodes))

    @property
    def nodes(self) -> Mapping[int, tuple[float, float]]:
        d = self._delta
        out = {n: p for n, p in self._base.nodes.items() if n not in d.remove_nodes}
        out.update(d.move_nodes)
        out.update(d.add_nodes)
        return out

    def __contains__(self, n: int) -> bool:
        d = self._delta
        if n in d.add_nodes:
            return True
        if n in d.remove_nodes:
            return False
        return n in self._base.node_ids

    def position(self, n: Any) -> tuple[float, float]:
        nid = _int_or_err(n, "node")
        if nid in self._delta.move_nodes:
            return self._delta.move_nodes[nid]
        if nid in self._delta.add_nodes:
            return self._delta.add_nodes[nid]
        if nid in self._delta.remove_nodes:
            raise StaleHandleError(f"node {nid} was removed by the staged edit")
        return self._base.position(nid)

    # -- cells -----------------------------------------------------------

    @property
    def cells(self) -> Mapping[int, tuple[int, int, ...]]:
        d = self._delta
        out = {c: b for c, b in self._base.cells.items() if c not in d.remove_cells}
        out.update(d.add_cells)
        return out

    def cell(self, cid: Any) -> tuple[int, int, ...]:
        c = _int_or_err(cid, "cell")
        if c in self._delta.add_cells:
            return self._delta.add_cells[c]
        if c in self._delta.remove_cells:
            raise StaleHandleError(f"cell {c} was removed by the staged edit")
        return self._base.cell(c)

    def cell_kind(self, cid: Any) -> str:
        c = _int_or_err(cid, "cell")
        if c in self._delta.add_cells:
            return self._delta.add_cell_kinds.get(c, "Q4")
        return self._base.cell_kind(c)

    # -- incidence / adjacency ------------------------------------------

    def edge_cells(self, key: Any) -> tuple[int, ...]:
        try:
            k = _edge_key(_int_or_err(key[0], "edge[0]"), _int_or_err(key[1], "edge[1]"))
        except (MeshError, StaleHandleError):
            return ()
        inc = set(self._base.edge_cells(k))
        net = self._touched().get(k)
        if net:
            for cid, sign in net.items():
                if sign > 0:
                    inc.add(cid)
                elif sign < 0:
                    inc.discard(cid)
        return tuple(sorted(inc))

    def neighbors_at(self, n: Any) -> tuple[EdgeKey, ...]:
        nid = _int_or_err(n, "node")
        if nid not in self:
            raise StaleHandleError(f"node {nid} is not live in the candidate")
        edges = set(self._base.neighbors_at(nid))
        for k in self._touched():
            if nid in k:
                if self.edge_cells(k):
                    edges.add(k)
                else:
                    edges.discard(k)
        return tuple(sorted(edges))

    def front_neighbors_at(self, n: Any, *, exclude: EdgeKey | None = None) -> tuple[EdgeKey, ...]:
        nid = _int_or_err(n, "node")
        if nid not in self:
            raise StaleHandleError(f"node {nid} is not live in the candidate")
        out = set(self._base.front_neighbors_at(nid, exclude=exclude))
        for k in self._delta.add_front:
            if nid in k and (exclude is None or k != exclude):
                out.add(k)
        for k in self._delta.remove_front:
            if nid in k:
                out.discard(k)
        return tuple(out)

    def cells_at(self, n: Any) -> tuple[int, ...]:
        nid = _int_or_err(n, "node")
        out = set(self._base.cells_at(nid))
        for cid, body in self._delta.add_cells.items():
            if nid in body:
                out.add(cid)
        out -= self._delta.remove_cells
        return tuple(sorted(out))

    def cells_at_edge(self, key: Any) -> tuple[int, ...]:
        return self.edge_cells(key)

    def other_node_of(self, key: Any, n: Any) -> int:
        return self._base.other_node_of(key, n)

    # -- front / bits ----------------------------------------------------

    def is_front_edge(self, key: Any) -> bool:
        try:
            k = _edge_key(_int_or_err(key[0], "edge[0]"), _int_or_err(key[1], "edge[1]"))
        except (MeshError, StaleHandleError):
            return False
        d = self._delta
        if k in d.add_front:
            return True
        if k in d.remove_front:
            return False
        return self._base.is_front_edge(k)

    def bit_set(self, edge: Sequence[int], n: Any) -> bool:
        k = _edge_key(_int_or_err(edge[0], "edge[0]"), _int_or_err(edge[1], "edge[1]"))
        nn = _int_or_err(n, "bit node")
        d = self._delta
        if (k, nn) in d.add_bits:
            return True
        if (k, nn) in d.remove_bits:
            return False
        return self._base.bit_set((k[0], k[1]), nn)

    @property
    def front(self) -> frozenset[EdgeKey]:
        d = self._delta
        return frozenset((self._base.front - d.remove_front) | d.add_front)

    @property
    def edges(self) -> frozenset[EdgeKey]:
        """All edges with at least one resident cell in the candidate."""
        candidates = set(self._base.edges)
        candidates |= set(self._touched())
        return frozenset(k for k in candidates if self.edge_cells(k))

    @property
    def front_bits(self) -> frozenset[tuple[EdgeKey, int]]:
        d = self._delta
        return frozenset((self._base.front_bits - d.remove_bits) | d.add_bits)

    # -- protected -------------------------------------------------------

    def is_protected_node(self, n: Any) -> bool:
        try:
            nn = _int_or_err(n, "protected node")
        except (MeshError, StaleHandleError):
            return False
        d = self._delta
        if nn in d.add_prot_nodes:
            return True
        if nn in d.remove_prot_nodes:
            return False
        return self._base.is_protected_node(nn)

    def is_protected_edge(self, key: Any) -> bool:
        try:
            k = _edge_key(_int_or_err(key[0], "edge[0]"), _int_or_err(key[1], "edge[1]"))
        except (MeshError, StaleHandleError):
            return False
        d = self._delta
        if k in d.add_prot_edges:
            return True
        if k in d.remove_prot_edges:
            return False
        return self._base.is_protected_edge(k)

    @property
    def protected_nodes(self) -> frozenset[int]:
        d = self._delta
        return frozenset((self._base.protected_nodes - d.remove_prot_nodes) | d.add_prot_nodes)

    @property
    def protected_edges(self) -> frozenset[EdgeKey]:
        d = self._delta
        return frozenset(
            (self._base.protected_edges - d.remove_prot_edges) | d.add_prot_edges
        )

    # -- misc ------------------------------------------------------------

    def is_cancelled(self) -> bool:
        return self._base.is_cancelled()

    def checkpoint(self) -> None:
        self._base.checkpoint()

    def digest(self) -> str:
        d = self._delta
        keys = set(d.touched_map(self._base))
        keys |= d.add_front | d.remove_front | d.add_prot_edges | d.remove_prot_edges
        keys |= {k for k, _n in d.add_bits}
        keys |= {k for k, _n in d.remove_bits}
        payload = {
            "generation": self._base.generation,
            "node_id_count": len(self.node_ids),
            "cells": {c: list(self.cell(c)) for c in sorted(self.cells)},
            "cell_kinds": {c: self.cell_kind(c) for c in sorted(self.cells)},
            "front": [[a, b] for a, b in sorted(self.front)],
            "front_bits": [[a, b, n] for (a, b), n in sorted(self.front_bits)],
            "protected_nodes": sorted(self.protected_nodes),
            "protected_edges": [[a, b] for a, b in sorted(self.protected_edges)],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
