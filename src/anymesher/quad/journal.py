"""Transactional publication for the quad-first advancing front.

A :class:`Transaction` is the *only* writer of a :class:`~.state.QuadMeshState`.
It stages a sparse :class:`~.state.Delta` against a captured resident base and
either

* ``commit()``s it — after guarding that the base generation has not moved and
  passing the cancellation checkpoint — in which case the base atomically
  applies the delta and bumps its generation, or
* ``rollback()``s it — which is **discard-only**: the staged delta is thrown
  away and the base is left byte-for-byte identical to the captured generation.

There is no replay and no full-mesh deep-copy; the staged edit is the sparse
:class:`Delta` and reads go through the thin :class:`~.state.View` proxy.

Usage::

    with base.transaction() as tx:
        tx.view            # read candidate base+delta (O(local))
        tx.add_cell(...)   # stage sparse edits
        tx.commit()        # atomic publish or raise and leave base untouched
"""

from __future__ import annotations

from typing import Any

from ..errors import MeshError
from .state import Delta, EdgeKey, QuadMeshState, StaleHandleError, _check_position, View

__all__ = ["Transaction", "TransactionStateError"]


class TransactionStateError(MeshError):
    """Raised when a transaction is used after it has concluded or was superseded."""


class Transaction:
    """Sparse edit-then-publish wrapper over a resident quad mesh state.

    Parameters
    ----------
    base:
        The resident :class:`QuadMeshState` this transaction will read and (on
        commit) write.  The transaction captures ``base.generation`` at
        construction; if any other writer has already committed a delta (so the
        base generation advanced), ``commit`` refuses with a
        :class:`TransactionStateError` rather than silently overwriting.
    """

    def __init__(self, base: QuadMeshState) -> None:
        self._base = base
        self._base_generation = base.generation
        self._delta = Delta()
        self._node_cursor = base.next_node_id
        self._cell_cursor = base.next_cell_id
        self._committed = False
        self._rolled_back = False
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle state
    # ------------------------------------------------------------------

    @property
    def base(self) -> QuadMeshState:
        return self._base

    @property
    def delta(self) -> Delta:
        return self._delta

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def rolled_back(self) -> bool:
        return self._rolled_back

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def view(self) -> View:
        """A fresh read-only candidate view of ``base + delta``."""
        self._ensure_open("read")
        return self._base.combine(self._delta)

    # ------------------------------------------------------------------
    # Cancellation checkpoint (delegates to the resident state)
    # ------------------------------------------------------------------

    def is_cancelled(self) -> bool:
        return self._base.is_cancelled()

    def checkpoint(self) -> None:
        """Raise if the caller has requested cancellation.

        Call this at meaningful granularity between staged edits so a long
        front advance can be interrupted without partial publication.
        """
        self._ensure_open("checkpoint")
        self._base.checkpoint()

    # ------------------------------------------------------------------
    # Thin staging helpers (forward to the sparse delta)
    # ------------------------------------------------------------------

    def add_node(self, node: int, position: Any) -> None:
        self._ensure_open("stage")
        self._delta.add_nodes[int(node)] = position

    def move_node(self, node: int, position: Any) -> None:
        """Stage a coordinate-only move of an existing, unprotected node."""
        self._ensure_open("stage")
        nid = _int(node)
        if nid in self._delta.add_nodes or nid in self._delta.remove_nodes:
            raise MeshError(f"node {nid} cannot be both moved and added/removed")
        if self._base.is_protected_node(nid):
            raise MeshError(f"node {nid} is protected and cannot be moved")
        try:
            self._base.position(nid)
        except StaleHandleError as exc:
            raise MeshError(f"node {nid} is not a live node in this state") from exc
        self._delta.move_nodes[nid] = _check_position(position)

    def allocate_node(self, position: Any) -> int:
        """Take the next node id for this transaction and stage it at ``position``."""
        self._ensure_open("stage")
        nid = self._node_cursor
        self._node_cursor += 1
        self._delta.add_nodes[nid] = _check_position(position)
        return nid

    def allocate_cell(self, body: Any, kind: str | None = None) -> int:
        """Take the next cell id for this transaction and stage the body."""
        self._ensure_open("stage")
        cid = self._cell_cursor
        self._cell_cursor += 1
        body_ids = tuple(_int(x) for x in body)
        if kind is None:
            kind = "Q4" if len(body_ids) == 4 else "T3"
        self._delta.add_cells[cid] = body_ids
        self._delta.add_cell_kinds[cid] = kind
        return cid

    def remove_node(self, node: int) -> None:
        self._ensure_open("stage")
        self._delta.remove_nodes.add(int(node))

    def add_cell(self, cell: int, body: Any, kind: str | None = None) -> None:
        self._ensure_open("stage")
        cid = int(cell)
        self._delta.add_cells[cid] = tuple(int(x) for x in body)
        if kind is not None:
            self._delta.add_cell_kinds[cid] = kind

    def remove_cell(self, cell: int) -> None:
        self._ensure_open("stage")
        self._delta.remove_cells.add(int(cell))

    def add_front_edge(self, a: int, b: int) -> None:
        self._ensure_open("stage")
        self._delta.add_front.add(_key(a, b))

    def remove_front_edge(self, a: int, b: int) -> None:
        self._ensure_open("stage")
        self._delta.remove_front.add(_key(a, b))

    def set_side_bit(self, edge: EdgeKey, node: int) -> None:
        self._ensure_open("stage")
        self._delta.add_bits.add((edge, int(node)))

    def clear_side_bit(self, edge: EdgeKey, node: int) -> None:
        self._ensure_open("stage")
        self._delta.remove_bits.add((edge, int(node)))

    def protect_node(self, node: int) -> None:
        self._ensure_open("stage")
        self._delta.add_prot_nodes.add(int(node))

    def release_node(self, node: int) -> None:
        self._ensure_open("stage")
        self._delta.remove_prot_nodes.add(int(node))

    def protect_edge(self, a: int, b: int) -> None:
        self._ensure_open("stage")
        self._delta.add_prot_edges.add(_key(a, b))

    def release_edge(self, a: int, b: int) -> None:
        self._ensure_open("stage")
        self._delta.remove_prot_edges.add(_key(a, b))

    # ------------------------------------------------------------------
    # Atomic publish / discard-only rollback
    # ------------------------------------------------------------------

    def commit(self) -> QuadMeshState:
        """Prevalidate then atomically publish the staged delta.

        Guard order: (1) transaction still open, (2) base generation has not
        moved since capture, (3) cancellation checkpoint, (4) base applies the
        delta (a single non-cancellable, all-or-nothing section).  Any failure
        leaves the base untouched at its captured generation.
        """
        self._ensure_open("commit")
        if self._base.generation != self._base_generation:
            raise TransactionStateError(
                "base state generation advanced under an open transaction "
                f"(expected {self._base_generation}, found {self._base.generation})"
            )
        if self._delta.is_empty():
            # Nothing staged: a no-op commit is allowed and still closes the tx.
            self._committed = True
            self._closed = True
            return self._base
        self._base.checkpoint()  # cancellation gate before mutating
        self._base._apply_delta(self._delta)  # prevalidates + applies atomically
        self._committed = True
        self._closed = True
        return self._base

    def rollback(self) -> None:
        """Discard the staged delta; the base is left exactly as captured."""
        self._ensure_open("rollback")
        self._delta = Delta()
        self._rolled_back = True
        self._closed = True

    def close(self) -> None:
        """Close without publishing.  If nothing has concluded yet, this is a rollback."""
        if not self._committed and not self._rolled_back:
            self._delta = Delta()
            self._rolled_back = True
        self._closed = True

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        # Never suppress exceptions.  On any exit that did not conclude with a
        # commit, discard the staged edit (discard-only rollback).
        if not self._committed and not self._rolled_back:
            self._delta = Delta()
            self._rolled_back = True
        self._closed = True
        return False

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _ensure_open(self, action: str) -> None:
        if self._closed:
            raise TransactionStateError(f"transaction already closed; cannot {action}")
        if self._committed:
            raise TransactionStateError("transaction already committed; cannot stage")
        if self._rolled_back:
            raise TransactionStateError("transaction already rolled back; cannot stage")


def _key(a: int, b: int) -> EdgeKey:
    if a == b:
        raise MeshError("edge must have two distinct node ids")
    return (a, b) if a < b else (b, a)


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeshError("node/cell handle must be an integer")
    return value
