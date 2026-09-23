"""Quad-first Q4: representable private patch/corridor count-system model.

A *count system* is a small, pure-integer, combinatorial transport problem the
quad-first programme is allowed to pose to the integer min-cost-flow worker.
It carries two disjoint groups of integer supply/demand *stations*, an
nonneg-integer cost matrix between them, and an optional set of *blocked*
arcs (a structural constraint that forbids a specific pair), and nothing else.

This module defines the domain object and the representability predicate; it
knows nothing about the worker, the adapter or the solver.  The typed
failure :class:`CountRejected` is the single, non-fatal rejection channel —
mirroring :class:`~anymesher.quad.front.FrontRejected` — and is the only path
through which an out-of-scope count system can be surfaced to a caller.

Hard constraints (never relaxed):

* ``supplies``, ``demands`` and every entry of ``cost`` must be nonnegative
  Python ints (``bool`` is rejected, as in the rest of the package).
* ``sum(supplies) == sum(demands)`` (conservation at the whole-instance level).
* Each supply station must have at least one *unblocked* arc to a demand
  station and vice versa; otherwise the instance is trivially infeasible and
  out of scope for the Q4 worker (it does not model "no feasible solution").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Collection, Mapping

from ..errors import MeshError

__all__ = ["CountRejected", "CountInstance"]


class CountRejected(MeshError):
    """A specific count-system representability rule rejects this instance.

    Typed and non-fatal: the caller is expected to inspect the message and
    decide how to proceed (skip, fall back to a legacy path, raise a higher-
    level error).  There is no silent fallback in this module.
    """


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise CountRejected(f"{name} must be a nonnegative integer (got {value!r})")
    return int(value)


def _strict_int(value: Any, name: str) -> int:
    """Strict integer check: the value must *already* be an ``Integral``.

    No coercion is performed — booleans, floats (even integral-valued ones),
    numeric strings, and any other convertible-but-not-``Integral`` value is
    rejected before an ``int()`` call is ever made, so the caller never sees
    a silently-normalized instance.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CountRejected(
            f"{name} must be an integer (got {value!r} of "
            f"type {type(value).__name__})"
        )
    return int(value)


@dataclass(frozen=True)
class CountInstance:
    """Frozen representable count-system instance (Q4 schema 1).

    ``supplies[i]`` is the integer supply at station ``i`` (``0 <= i <
    len(supplies)``); ``demands[j]`` is the integer demand at station ``j``.
    ``cost[i][j]`` is the nonnegative integer cost of unit flow from station
    ``i`` to ``j``.  ``blocked`` is the (possibly empty) set of ``(i, j)``
    arcs that are *forbidden* by a structural constraint.  The instance is
    considered representable iff :func:`representable` (or equivalently
    :meth:`validate`) returns / does not raise.
    """

    supplies: tuple[int, ...]
    demands: tuple[int, ...]
    cost: tuple[tuple[int, ...], ...]            # cost[i][j]
    blocked: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        supplies: Collection[Any],
        demands: Collection[Any],
        cost: Mapping[Any, Mapping[Any, Any]] | Collection[Collection[Any]]
        | Mapping[Any, Any],
        blocked: Collection[tuple[int, int]] = (),
    ) -> "CountInstance":
        """Build from a collection of *strict* integers (lists, tuples, numpy).

        Every supply, demand, cost entry and blocked-arc endpoint is checked
        as an :class:`numbers.Integral` **before any coercion is attempted**;
        booleans, floats (even integral-valued), numeric strings and any other
        convertible-but-not-``Integral`` value is rejected with
        :class:`CountRejected`.  A *mapping* form for ``cost`` (``cost[i][j]``)
        is normalized to a tuple-of-tuples so the canonical form is
        unambiguous.
        """
        s = tuple(
            _strict_int(x, f"supplies[{i}]") for i, x in enumerate(supplies)
        )
        d = tuple(
            _strict_int(x, f"demands[{i}]") for i, x in enumerate(demands)
        )
        if isinstance(cost, Mapping):
            matrix = tuple(
                tuple(_strict_int(cost[i][j], f"cost[{i}][{j}]") for j in range(len(d)))
                for i in range(len(s))
            )
        else:
            rows = tuple(cost)
            matrix = tuple(
                tuple(
                    _strict_int(x, f"cost[{i}][{j}]") for j, x in enumerate(row)
                )
                for i, row in enumerate(rows)
            )
        blocked_set = frozenset(
            (
                _strict_int(b[0], f"blocked[{k}][0]"),
                _strict_int(b[1], f"blocked[{k}][1]"),
            )
            for k, b in enumerate(blocked)
        )
        return cls(
            supplies=s,
            demands=d,
            cost=matrix,
            blocked=blocked_set,
        )

    # ------------------------------------------------------------------
    # Representability
    # ------------------------------------------------------------------

    @property
    def n_in(self) -> int:
        return len(self.supplies)

    @property
    def n_out(self) -> int:
        return len(self.demands)

    @property
    def total_supply(self) -> int:
        return sum(self.supplies)

    @property
    def total_demand(self) -> int:
        return sum(self.demands)

    def is_block(self, i: int, j: int) -> bool:
        return (i, j) in self.blocked

    def unblocked(self) -> tuple[tuple[int, int], ...]:
        """All admissible arcs (supplies index × demands index) minus blocked."""
        return tuple(
            (i, j)
            for i in range(self.n_in)
            for j in range(self.n_out)
            if not self.is_block(i, j)
        )

    # ------------------------------------------------------------------
    # Validation (typed)
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`CountRejected` on any representability violation.

        The checks are deliberately strict: the Q4 worker is a pure min-cost-
        flow solver and does not model the "no feasible solution" case, so an
        instance with an isolated supply or demand station is *out of scope*
        for it and is rejected by this predicate instead.
        """
        s = self.supplies
        d = self.demands
        c = self.cost

        for i, v in enumerate(s):
            _nonneg_int(v, f"supplies[{i}]")
        for j, v in enumerate(d):
            _nonneg_int(v, f"demands[{j}]")
        for i, row in enumerate(c):
            if len(row) != len(d):
                raise CountRejected(
                    f"len(cost[{i}])={len(row)} != len(demands)={len(d)}"
                )
            for j, v in enumerate(row):
                _nonneg_int(v, f"cost[{i}][{j}]")

        if self.total_supply != self.total_demand:
            raise CountRejected(
                f"total supply {self.total_supply} != total demand {self.total_demand}"
            )

        for i, v in enumerate(s):
            if v > 0:
                if not any(not self.is_block(i, j) for j in range(self.n_out)):
                    raise CountRejected(f"supplies[{i}]={v} has no unblocked demand")
        for j, v in enumerate(d):
            if v > 0:
                if not any(not self.is_block(i, j) for i in range(self.n_in)):
                    raise CountRejected(f"demands[{j}]={v} has no unblocked supply")

    # ------------------------------------------------------------------
    # Serialization (for pinning / logging — NOT for the worker)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "anymesher.quad-count/1",
            "supplies": list(self.supplies),
            "demands": list(self.demands),
            "cost": [list(row) for row in self.cost],
            "blocked": sorted(tuple(b) for b in self.blocked),
        }
