"""Edge seeding: how many elements each line is divided into.

Under mapped meshing this is the piece with no fallback, so it gets solved
explicitly rather than guessed at.

Every edge carries a division count ``n``.  A mapped face requires its
opposite sides to agree::

    sum(n over side 0) == sum(n over side 2)
    sum(n over side 1) == sum(n over side 3)

A shared edge has exactly one ``n``, which is where persistent topology pays
off: conformity between neighbouring faces becomes automatic rather than a
coincident-node tolerance problem.  But the constraints **propagate across the
whole assembly** -- seeding one edge forces its opposite, which forces the
neighbouring face's edge across the shared boundary, and so on.

The solve is:

1. derive a desired ``n`` per edge by integrating the size field along it --
   just ``length / target`` when the field is uniform -- or take an explicit
   override;
2. union-find the forced equalities -- opposite sides that are both a single
   edge must have equal ``n`` -- into equivalence classes holding one ``n``;
3. bounded iterative repair for multi-edge chains: raise the coarser side to
   match the finer, refining the coarsest edges first;
4. re-verify, and report any unresolvable configuration by naming the face.

Repair only ever refines, so the process is monotone.  Pathological cyclic
topologies are caught by a sweep cap rather than looping forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from anygeometry.entities import Face, OrientedEdge
from anygeometry.model import GeometryModel

from .refinement import SizeField

__all__ = [
    "Seeding",
    "SeedingConflict",
    "edge_demand",
    "edge_distribution",
    "solve_seeding",
]

# Samples per edge when integrating a size field.  The field varies smoothly
# and only its integral is wanted, so this is plenty; it is fixed rather than
# adaptive so two runs of the same model always seed identically.
_FIELD_SAMPLES = 64


class SeedingConflict(ValueError):
    """Raised when no division count satisfies every mapped-face constraint."""


@dataclass
class Seeding:
    """Division counts per edge, satisfying every mapped-face constraint.

    Carries the size field it was solved against so node *placement* grades the
    same way the counts did.  A seeding solved for one field and used to place
    nodes against another would put the right number of nodes in the wrong
    places.
    """

    divisions: Dict[int, int]
    sweeps: int = 0
    classes: Dict[int, int] = field(default_factory=dict)
    size_field: SizeField | None = None

    def __getitem__(self, edge_id: int) -> int:
        return self.divisions[edge_id]

    def side_divisions(self, side: Sequence[OrientedEdge]) -> int:
        return sum(self.divisions[item.edge] for item in side)

    def total_elements_estimate(self, geometry: GeometryModel) -> int:
        """Shell element count the current seeding would produce."""

        total = 0
        for face in geometry.faces.values():
            sides = face.sides()
            total += self.side_divisions(sides[0]) * self.side_divisions(sides[1])
        return total


def edge_demand(
    geometry: GeometryModel, edge_id: int, size_field: SizeField
) -> float:
    """How many elements a size field asks for along one edge.

    The integral of ``ds / size(s)``: with a uniform field this is just
    ``length / target``, and with a graded one it counts each stretch of the
    edge against the size wanted *there*.  Returned unrounded so the caller
    decides how to turn it into a whole number of divisions.
    """

    if size_field.is_uniform:
        return geometry.edge_length(edge_id) / size_field.target_size

    parameters = np.linspace(0.0, 1.0, _FIELD_SAMPLES + 1)
    points = geometry.sample_edge(edge_id, parameters)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Size at each segment midpoint, which is where the segment is spent.
    midpoints = 0.5 * (points[:-1] + points[1:])
    sizes = size_field.size_at(midpoints)
    return float(np.sum(steps / sizes))


def edge_distribution(
    geometry: GeometryModel, edge_id: int, divisions: int, size_field: SizeField
) -> np.ndarray:
    """Interior node parameters along an edge, graded by the size field.

    Nodes are placed at equal spacing in the *unit mesh* -- the coordinate in
    which one unit is one element -- so a stretch asking for half the size gets
    twice the nodes.  With a uniform field this reduces to equal spacing in
    arc length, and for a straight line or an arc that is equal spacing in
    parameter, which is exactly what the ungraded mesher already did.
    """

    if divisions < 2:
        return np.zeros(0, dtype=float)

    parameters = np.linspace(0.0, 1.0, _FIELD_SAMPLES + 1)
    points = geometry.sample_edge(edge_id, parameters)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    midpoints = 0.5 * (points[:-1] + points[1:])
    demand = steps / size_field.size_at(midpoints)
    cumulative = np.concatenate(([0.0], np.cumsum(demand)))
    total = float(cumulative[-1])
    if total <= 0.0:
        raise SeedingConflict(f"edge {edge_id} has no length to divide")

    wanted = np.arange(1, divisions, dtype=float) * total / divisions
    # Monotone, so a plain interpolation inverts it.
    return np.interp(wanted, cumulative, parameters)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}

    def add(self, item: int) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: int) -> int:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Keep the lower ID as the representative so results are stable.
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            self._parent[right_root] = left_root


def solve_seeding(
    geometry: GeometryModel,
    *,
    target_size: float | None = None,
    size_field: SizeField | None = None,
    overrides: Mapping[int, int] | None = None,
    minimums: Mapping[int, int] | None = None,
    edge_ids: Iterable[int] | None = None,
    max_sweeps: int = 200,
    max_divisions: int = 100_000,
) -> Seeding:
    """Resolve division counts for every edge.

    ``target_size`` is the desired element edge length; pass a ``size_field``
    instead to seed against local refinement zones.  ``overrides`` pins
    specific edges; pinned edges are never refined, so a pinned edge that
    conflicts with another pinned edge is reported instead of being silently
    overridden.  ``minimums`` carries non-locking planner lower bounds, so a
    later local size field can still refine an accepted structured layout.
    """

    if size_field is None:
        if target_size is None:
            raise ValueError("solve_seeding needs a target_size or a size_field")
        size_field = SizeField(geometry, target_size)
    elif target_size is not None and target_size != size_field.target_size:
        raise ValueError(
            "solve_seeding was given both a target_size and a size_field with "
            "a different target; pass one or the other"
        )

    overrides = dict(overrides or {})
    minimums = dict(minimums or {})
    edges = (
        list(geometry.edges)
        if edge_ids is None
        else list(dict.fromkeys(int(e) for e in edge_ids))
    )
    for label, values in (("override", overrides), ("minimum", minimums)):
        for edge_id, divisions in values.items():
            if edge_id not in geometry.edges:
                raise SeedingConflict(
                    f"{label} given for unknown edge {edge_id}"
                )
            if isinstance(divisions, bool) or int(divisions) < 1:
                raise SeedingConflict(
                    f"edge {edge_id} {label} must be at least 1 division"
                )
            values[edge_id] = int(divisions)

    desired = {
        edge_id: max(
            1,
            int(round(edge_demand(geometry, edge_id, size_field))),
            minimums.get(edge_id, 1),
        )
        for edge_id in edges
    }

    faces = [
        face
        for face in geometry.faces.values()
        if len(face.corners) == 4
        and not face.holes
        and all(item.edge in desired for item in face.loop)
    ]

    union = _UnionFind()
    for edge_id in edges:
        union.add(edge_id)
    for face in faces:
        sides = face.sides()
        for first, second in ((0, 2), (1, 3)):
            if len(sides[first]) == 1 and len(sides[second]) == 1:
                union.union(sides[first][0].edge, sides[second][0].edge)

    members: Dict[int, List[int]] = {}
    for edge_id in edges:
        members.setdefault(union.find(edge_id), []).append(edge_id)

    counts: Dict[int, int] = {}
    locked: Dict[int, bool] = {}
    for root, group in members.items():
        pinned = {
            overrides[edge_id] for edge_id in group if edge_id in overrides
        }
        if len(pinned) > 1:
            raise SeedingConflict(
                "conflicting division overrides on edges "
                f"{sorted(e for e in group if e in overrides)}: "
                f"{sorted(pinned)}. These edges are forced to share a division "
                "count by opposite-side constraints."
            )
        if pinned:
            pinned_count = pinned.pop()
            minimum_count = max(minimums.get(edge_id, 1) for edge_id in group)
            if pinned_count < minimum_count:
                raise SeedingConflict(
                    "division override is below the non-locking planner minimum "
                    f"for edges {sorted(group)}: {pinned_count} < {minimum_count}"
                )
            counts[root] = pinned_count
            locked[root] = True
        else:
            # Err finer: taking the max never under-refines a shared edge.
            counts[root] = max(desired[edge_id] for edge_id in group)
            locked[root] = False

    # Repair ranks edges by how many elements the field wants along them, not
    # by raw length, so grading steers refinement to where it is asked for.
    # Under a uniform field this is length divided by a constant, so the
    # ordering -- and therefore the resulting mesh -- is unchanged.
    demands = {edge_id: desired[edge_id] for edge_id in edges}

    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        changed = False
        for face in faces:
            sides = face.sides()
            for first, second in ((0, 2), (1, 3)):
                if _repair_axis(
                    face,
                    sides[first],
                    sides[second],
                    union=union,
                    counts=counts,
                    locked=locked,
                    demands=demands,
                    max_divisions=max_divisions,
                ):
                    changed = True
        if not changed:
            break
    else:
        raise SeedingConflict(
            f"edge seeding did not settle within {max_sweeps} sweeps; the "
            "faces form a constraint cycle that keeps forcing refinement. "
            "Pin division counts with overrides, or split the offending faces."
        )

    divisions = {edge_id: counts[union.find(edge_id)] for edge_id in edges}
    _verify(geometry, faces, divisions)
    return Seeding(
        divisions=divisions,
        sweeps=sweeps,
        classes={edge_id: union.find(edge_id) for edge_id in edges},
        size_field=size_field,
    )


def _repair_axis(
    face: Face,
    side_a: Sequence[OrientedEdge],
    side_b: Sequence[OrientedEdge],
    *,
    union: _UnionFind,
    counts: Dict[int, int],
    locked: Dict[int, bool],
    demands: Mapping[int, float],
    max_divisions: int,
) -> bool:
    """Bring one pair of opposite sides into agreement.  True if changed."""

    total_a = sum(counts[union.find(item.edge)] for item in side_a)
    total_b = sum(counts[union.find(item.edge)] for item in side_b)
    if total_a == total_b:
        return False

    short = side_a if total_a < total_b else side_b
    deficit = abs(total_a - total_b)

    # A class can appear more than once on the same side; adding one division
    # to it then raises the side total by its multiplicity.
    weights: Dict[int, int] = {}
    class_demand: Dict[int, float] = {}
    for item in short:
        root = union.find(item.edge)
        weights[root] = weights.get(root, 0) + 1
        class_demand[root] = class_demand.get(root, 0.0) + demands[item.edge]

    adjustable = [root for root in weights if not locked[root]]
    if not adjustable:
        raise SeedingConflict(
            f"face {face.id} needs {deficit} more division(s) on one side, but "
            "every edge there has a pinned division count. Relax an override "
            "or split the face."
        )

    while deficit > 0:
        candidates = [
            root for root in adjustable if weights[root] <= deficit
        ]
        if not candidates:
            raise SeedingConflict(
                f"face {face.id} cannot match its opposite sides exactly: "
                f"{deficit} division(s) remain but every adjustable edge there "
                "would overshoot. Split the face or pin its divisions."
            )
        # Refine the coarsest edge first, so repair improves mesh quality
        # rather than degrading it.  Ties break on class ID for determinism.
        chosen = max(
            candidates,
            key=lambda root: (class_demand[root] / counts[root], -root),
        )
        counts[chosen] += 1
        if counts[chosen] > max_divisions:
            raise SeedingConflict(
                f"edge seeding exceeded {max_divisions} divisions on one edge "
                f"while repairing face {face.id}; the model is over-constrained."
            )
        deficit -= weights[chosen]

    return True


def _verify(
    geometry: GeometryModel,
    faces: Sequence[Face],
    divisions: Mapping[int, int],
) -> None:
    for face in faces:
        sides = face.sides()
        for first, second in ((0, 2), (1, 3)):
            total_a = sum(divisions[item.edge] for item in sides[first])
            total_b = sum(divisions[item.edge] for item in sides[second])
            if total_a != total_b:
                raise SeedingConflict(
                    f"face {face.id} opposite sides disagree after seeding: "
                    f"{total_a} vs {total_b} divisions"
                )
