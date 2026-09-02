"""Quality-ranked, deterministic T3-to-Q4 recombination."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from numbers import Integral
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .core import MeshCore, corner_edges
from .errors import MeshError
from .quality_v2 import quad_candidate_quality
from .native_cpp import (
    COMPILED_QUALITY_PIPELINE_AVAILABLE,
    native_recombination_decisions,
)

__all__ = [
    "RecombinationReport",
    "pair_triangles",
    "recombine_to_quads",
    "recombine_triangles",
    "recombine_triangles_with_report",
]


@dataclass(frozen=True)
class _Candidate:
    first: int
    second: int
    shared: tuple[int, int]
    corners: tuple[int, int, int, int]
    score: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class RecombinationReport:
    mesh: MeshCore
    paired_triangle_ids: np.ndarray
    created_quad_ids: np.ndarray
    scores: np.ndarray
    candidate_count: int
    rejected_candidate_count: int
    exchange_count: int = 0
    exchange_work: int = 0
    exchange_truncated: bool = False

    @property
    def pair_count(self) -> int:
        return int(len(self.created_quad_ids))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mesh, name)


def _edge_incidence(mesh: MeshCore) -> dict[tuple[int, int], list[int]]:
    incidence: dict[tuple[int, int], list[int]] = {}
    for row in np.flatnonzero(mesh.triangle_active):
        for edge in corner_edges(mesh.triangle_connectivity[row]):
            incidence.setdefault(edge, []).append(int(row))
    return incidence


def _quad_cycle(mesh: MeshCore, first_row: int, second_row: int) -> tuple[int, int, int, int] | None:
    first = mesh.triangle_connectivity[first_row, :3]
    second = mesh.triangle_connectivity[second_row, :3]
    counts: dict[tuple[int, int], int] = {}
    for triangle in (first, second):
        for edge in corner_edges(triangle):
            counts[edge] = counts.get(edge, 0) + 1
    boundary = [edge for edge, count in counts.items() if count == 1]
    if len(boundary) != 4:
        return None
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if len(adjacency) != 4 or any(len(values) != 2 for values in adjacency.values()):
        return None
    start = min(adjacency, key=lambda row: (int(mesh.node_ids[row]), row))
    options: list[tuple[int, int, int, int]] = []
    for next_node in sorted(adjacency[start], key=lambda row: (int(mesh.node_ids[row]), row)):
        cycle = [start, next_node]
        while len(cycle) < 4:
            candidates = [node for node in adjacency[cycle[-1]] if node != cycle[-2]]
            if not candidates:
                break
            cycle.append(candidates[0])
        if len(cycle) == 4 and start in adjacency[cycle[-1]] and len(set(cycle)) == 4:
            options.append(tuple(cycle))  # type: ignore[arg-type]
    if not options:
        return None
    reference = np.cross(
        mesh.node_coordinates[first[1]] - mesh.node_coordinates[first[0]],
        mesh.node_coordinates[first[2]] - mesh.node_coordinates[first[0]],
    ) + np.cross(
        mesh.node_coordinates[second[1]] - mesh.node_coordinates[second[0]],
        mesh.node_coordinates[second[2]] - mesh.node_coordinates[second[0]],
    )

    def aligned(cycle: tuple[int, int, int, int]) -> bool:
        coords = mesh.node_coordinates[np.asarray(cycle)]
        normal = sum(
            (np.cross(coords[index], coords[(index + 1) % 4]) for index in range(4)),
            start=np.zeros(3),
        )
        return float(np.dot(normal, reference)) >= 0.0

    aligned_options = [cycle for cycle in options if aligned(cycle)]
    pool = aligned_options or options
    return min(pool, key=lambda cycle: tuple(int(mesh.node_ids[row]) for row in cycle))


def _protected_rows(
    mesh: MeshCore,
    protected_edges: Iterable[Sequence[int]],
    edges_are_ids: bool,
) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for raw in protected_edges:
        if len(raw) != 2:
            raise MeshError("a protected edge needs two nodes")
        if edges_are_ids:
            first = mesh._row_for_id("node", int(raw[0]))
            second = mesh._row_for_id("node", int(raw[1]))
        else:
            first, second = int(raw[0]), int(raw[1])
            if min(first, second) < 0 or max(first, second) >= mesh.num_nodes:
                raise MeshError("protected edge references an invalid node row")
        result.add((min(first, second), max(first, second)))
    return result


def _publish_native_recombination(
    mesh: MeshCore,
    selected: Sequence[_Candidate],
    *,
    candidate_count: int,
    rejected_candidate_count: int,
    exchange_count: int,
    exchange_work: int,
    exchange_truncated: bool,
    cancellation_check: Callable[[str], None] | None,
) -> RecombinationReport:
    triangle_active = np.array(mesh.triangle_active, copy=True)
    for candidate in selected:
        triangle_active[candidate.first] = False
        triangle_active[candidate.second] = False

    quadratic = mesh.triangle_connectivity.shape[1] == 6
    new_connectivity: list[tuple[int, ...]] = []
    new_owners: list[int] = []
    for candidate in selected:
        if quadratic:
            edge_mid: dict[tuple[int, int], int] = {}
            for triangle_row in (candidate.first, candidate.second):
                connectivity = mesh.triangle_connectivity[triangle_row]
                for index, edge in enumerate(corner_edges(connectivity)):
                    edge_mid[edge] = int(connectivity[3 + index])
            mids = [
                edge_mid[(min(a, b), max(a, b))]
                for a, b in zip(
                    candidate.corners,
                    candidate.corners[1:] + candidate.corners[:1],
                )
            ]
            new_connectivity.append((*candidate.corners, *mids))
        else:
            new_connectivity.append(candidate.corners)
        first_owner = int(mesh.triangle_owner_handles[candidate.first])
        second_owner = int(mesh.triangle_owner_handles[candidate.second])
        new_owners.append(first_owner if first_owner == second_owner else -1)

    width = 8 if quadratic else 4
    appended = np.asarray(new_connectivity, dtype=np.int64).reshape((-1, width))
    if mesh.num_quads and mesh.quad_connectivity.shape[1] != width:
        raise MeshError("existing quadrilaterals use a different polynomial order")
    quad_connectivity = np.vstack((mesh.quad_connectivity, appended))
    maximum_id = int(np.max(mesh.element_ids)) if mesh.element_ids.size else 0
    created_ids = np.arange(
        maximum_id + 1,
        maximum_id + 1 + len(selected),
        dtype=np.int64,
    )
    quad_ids = np.concatenate((mesh.quad_ids, created_ids))
    quad_owners = np.concatenate(
        (mesh.quad_owner_handles, np.asarray(new_owners, dtype=np.int32))
    )
    quad_active = np.concatenate(
        (mesh.quad_active, np.ones(len(selected), dtype=bool))
    )
    result = MeshCore(
        mesh.node_coordinates,
        mesh.triangle_connectivity,
        quad_connectivity,
        node_ids=mesh.node_ids,
        triangle_ids=mesh.triangle_ids,
        quad_ids=quad_ids,
        owner_table=mesh.owner_table,
        node_owner_handles=mesh.node_owner_handles,
        triangle_owner_handles=mesh.triangle_owner_handles,
        quad_owner_handles=quad_owners,
        node_active=mesh.node_active,
        triangle_active=triangle_active,
        quad_active=quad_active,
    )
    pairs = np.asarray(
        [
            (
                int(mesh.triangle_ids[item.first]),
                int(mesh.triangle_ids[item.second]),
            )
            for item in selected
        ],
        dtype=np.int64,
    ).reshape((-1, 2))
    scores = np.asarray([item.score for item in selected], dtype=np.float64)
    pairs.setflags(write=False)
    created_ids.setflags(write=False)
    scores.setflags(write=False)
    if cancellation_check is not None:
        cancellation_check("triangle recombination complete")
    return RecombinationReport(
        result,
        pairs,
        created_ids,
        scores,
        candidate_count,
        rejected_candidate_count,
        exchange_count,
        exchange_work,
        exchange_truncated,
    )


def recombine_triangles_with_report(
    mesh_or_points: MeshCore | Any,
    triangles: Any | None = None,
    *,
    protected_edges: Iterable[Sequence[int]] = (),
    constraints: Iterable[Sequence[int]] | None = None,
    edges_are_ids: bool = False,
    min_scaled_jacobian: float = 0.20,
    max_aspect_ratio: float = 4.0,
    min_angle: float = 30.0,
    max_angle: float = 150.0,
    max_warpage: float = 0.10,
    cancellation_check: Callable[[str], None] | None = None,
    max_exchange_work: int = 1_000_000,
    _use_native: bool | None = None,
) -> RecombinationReport:
    """Pair adjacent triangles greedily from best quality to worst.

    A candidate which misses any hard threshold is left as two triangles.  The
    deterministic ordering uses quality first, then stable triangle/node IDs.
    Paired triangles remain in the dense table with inactive flags, preserving
    their IDs for diagnostics and incremental clients.
    """

    if (
        isinstance(max_exchange_work, bool)
        or not isinstance(max_exchange_work, Integral)
        or int(max_exchange_work) < 1
    ):
        raise MeshError("max_exchange_work must be a positive integer")
    max_exchange_work = int(max_exchange_work)
    if cancellation_check is not None:
        cancellation_check("triangle recombination start")

    if isinstance(mesh_or_points, MeshCore):
        if triangles is not None:
            raise MeshError("triangles are already part of MeshCore")
        mesh = mesh_or_points
    else:
        if triangles is None:
            raise MeshError("raw points need triangle connectivity")
        mesh = MeshCore(mesh_or_points, triangles)
    if mesh.triangle_connectivity.shape[1] not in (3, 6):
        raise MeshError("only T3/T6 triangles can be recombined")
    supplied_edges = protected_edges if constraints is None else constraints
    protected = _protected_rows(mesh, supplied_edges, edges_are_ids)
    native = None
    if _use_native is not False:
        if (
            COMPILED_QUALITY_PIPELINE_AVAILABLE
            and cancellation_check is not None
        ):
            cancellation_check("triangle recombination candidates")
        native = native_recombination_decisions(
            mesh.node_coordinates,
            np.ascontiguousarray(mesh.triangle_connectivity[:, :3], dtype=np.int64),
            mesh.triangle_ids,
            mesh.node_ids,
            np.ascontiguousarray(np.flatnonzero(mesh.triangle_active), dtype=np.int64),
            np.asarray(sorted(protected), dtype=np.int64).reshape((-1, 2)),
            min_scaled_jacobian=min_scaled_jacobian,
            max_aspect_ratio=max_aspect_ratio,
            min_angle=min_angle,
            max_angle=max_angle,
            max_warpage=max_warpage,
            max_exchange_work=max_exchange_work,
        )
    if native is not None:
        selected = [
            _Candidate(
                int(row[0]),
                int(row[1]),
                (int(row[2]), int(row[3])),
                tuple(int(value) for value in row[4:8]),
                float(row[8]),
                dict(
                    zip(
                        (
                            "area",
                            "aspect_ratio",
                            "minimum_angle",
                            "maximum_angle",
                            "scaled_jacobian",
                            "warpage",
                        ),
                        (float(value) for value in row[9:15]),
                    )
                ),
            )
            for row in native["selected"]
        ]
        return _publish_native_recombination(
            mesh,
            selected,
            candidate_count=int(native["candidate_count"]),
            rejected_candidate_count=int(native["rejected_candidate_count"]),
            exchange_count=int(native["exchange_count"]),
            exchange_work=int(native["exchange_work"]),
            exchange_truncated=bool(native["exchange_truncated"]),
            cancellation_check=cancellation_check,
        )
    incidence = _edge_incidence(mesh)
    candidates: list[_Candidate] = []
    rejected = 0
    for candidate_index, (edge, attached) in enumerate(sorted(incidence.items())):
        if cancellation_check is not None and candidate_index % 4096 == 0:
            cancellation_check("triangle recombination candidates")
        if len(attached) != 2 or edge in protected:
            continue
        first, second = attached
        corners = _quad_cycle(mesh, first, second)
        if corners is None:
            rejected += 1
            continue
        metrics = quad_candidate_quality(mesh.node_coordinates, corners)
        acceptable = (
            np.isfinite(list(metrics.values())).all()
            and metrics["scaled_jacobian"] >= min_scaled_jacobian
            and metrics["aspect_ratio"] <= max_aspect_ratio
            and metrics["minimum_angle"] >= min_angle
            and metrics["maximum_angle"] <= max_angle
            and metrics["warpage"] <= max_warpage
        )
        if not acceptable:
            rejected += 1
            continue
        score = (
            metrics["scaled_jacobian"]
            - 0.08 * log(max(metrics["aspect_ratio"], 1.0))
            - 0.002 * max(0.0, 90.0 - metrics["minimum_angle"])
            - 0.002 * max(0.0, metrics["maximum_angle"] - 90.0)
            - 0.25 * metrics["warpage"]
        )
        candidates.append(_Candidate(first, second, edge, corners, score, metrics))

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            min(int(mesh.triangle_ids[candidate.first]), int(mesh.triangle_ids[candidate.second])),
            max(int(mesh.triangle_ids[candidate.first]), int(mesh.triangle_ids[candidate.second])),
            tuple(int(mesh.node_ids[row]) for row in candidate.corners),
        )
    )
    used: set[int] = set()
    selected: list[_Candidate] = []
    for candidate in candidates:
        if candidate.first in used or candidate.second in used:
            continue
        used.update((candidate.first, candidate.second))
        selected.append(candidate)

    # A quality-first greedy matching can strand two otherwise pairable
    # triangles.  Repair that local case with a deterministic 1-for-2
    # exchange.  Candidate lookup is indexed by triangle row, so the repair
    # does not repeatedly scan the complete dual graph.  Each accepted
    # exchange consumes two previously free triangles and therefore the queue
    # and work counter are both bounded independently of mesh geometry.
    candidates_by_triangle: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        candidates_by_triangle.setdefault(candidate.first, []).append(candidate)
        candidates_by_triangle.setdefault(candidate.second, []).append(candidate)

    def candidate_key(candidate: _Candidate) -> tuple[int, int, tuple[int, int]]:
        return (candidate.first, candidate.second, candidate.shared)

    def other_row(candidate: _Candidate, row: int) -> int:
        return candidate.second if candidate.first == row else candidate.first

    active = {candidate_key(candidate): candidate for candidate in selected}
    queue = list(selected)
    exchange_count = 0
    exchange_work = 0
    exchange_truncated = False
    queue_index = 0
    while queue_index < len(queue):
        current = queue[queue_index]
        queue_index += 1
        current_key = candidate_key(current)
        if current_key not in active:
            continue
        first_options = [
            candidate
            for candidate in candidates_by_triangle.get(current.first, ())
            if candidate_key(candidate) != current_key
            and other_row(candidate, current.first) not in used
        ]
        second_options = [
            candidate
            for candidate in candidates_by_triangle.get(current.second, ())
            if candidate_key(candidate) != current_key
            and other_row(candidate, current.second) not in used
        ]
        possible: list[tuple[_Candidate, _Candidate, int, int]] = []
        stop = False
        for first in first_options:
            first_other = other_row(first, current.first)
            for second in second_options:
                if exchange_work >= max_exchange_work:
                    exchange_truncated = True
                    stop = True
                    break
                exchange_work += 1
                second_other = other_row(second, current.second)
                if first_other != second_other and first is not second:
                    possible.append((first, second, first_other, second_other))
            if stop:
                break
        if stop:
            break
        if not possible:
            continue

        def exchange_key(
            value: tuple[_Candidate, _Candidate, int, int]
        ) -> tuple[float, tuple[int, ...]]:
            first, second, first_other, second_other = value
            stable = tuple(
                sorted(
                    (
                        int(mesh.triangle_ids[current.first]),
                        int(mesh.triangle_ids[current.second]),
                        int(mesh.triangle_ids[first_other]),
                        int(mesh.triangle_ids[second_other]),
                    )
                )
            )
            return (-(first.score + second.score), stable)

        first, second, first_other, second_other = min(
            possible, key=exchange_key
        )
        active.pop(current_key)
        active[candidate_key(first)] = first
        active[candidate_key(second)] = second
        used.update((first_other, second_other))
        queue.extend((first, second))
        exchange_count += 1
        if cancellation_check is not None and exchange_count % 256 == 0:
            cancellation_check("triangle recombination exchange")

    selected = sorted(
        active.values(),
        key=lambda candidate: (
            min(
                int(mesh.triangle_ids[candidate.first]),
                int(mesh.triangle_ids[candidate.second]),
            ),
            max(
                int(mesh.triangle_ids[candidate.first]),
                int(mesh.triangle_ids[candidate.second]),
            ),
            tuple(int(mesh.node_ids[row]) for row in candidate.corners),
        ),
    )

    triangle_active = np.array(mesh.triangle_active, copy=True)
    for candidate in selected:
        triangle_active[candidate.first] = False
        triangle_active[candidate.second] = False

    quadratic = mesh.triangle_connectivity.shape[1] == 6
    new_connectivity: list[tuple[int, ...]] = []
    new_owners: list[int] = []
    for candidate in selected:
        if quadratic:
            edge_mid: dict[tuple[int, int], int] = {}
            for triangle_row in (candidate.first, candidate.second):
                connectivity = mesh.triangle_connectivity[triangle_row]
                for index, edge in enumerate(corner_edges(connectivity)):
                    edge_mid[edge] = int(connectivity[3 + index])
            mids = [edge_mid[(min(a, b), max(a, b))] for a, b in zip(candidate.corners, candidate.corners[1:] + candidate.corners[:1])]
            new_connectivity.append((*candidate.corners, *mids))
        else:
            new_connectivity.append(candidate.corners)
        first_owner = int(mesh.triangle_owner_handles[candidate.first])
        second_owner = int(mesh.triangle_owner_handles[candidate.second])
        new_owners.append(first_owner if first_owner == second_owner else -1)

    width = 8 if quadratic else 4
    appended = np.asarray(new_connectivity, dtype=np.int64).reshape((-1, width))
    if mesh.num_quads and mesh.quad_connectivity.shape[1] != width:
        raise MeshError("existing quadrilaterals use a different polynomial order")
    quad_connectivity = np.vstack((mesh.quad_connectivity, appended))
    maximum_id = int(np.max(mesh.element_ids)) if mesh.element_ids.size else 0
    created_ids = np.arange(maximum_id + 1, maximum_id + 1 + len(selected), dtype=np.int64)
    quad_ids = np.concatenate((mesh.quad_ids, created_ids))
    quad_owners = np.concatenate((mesh.quad_owner_handles, np.asarray(new_owners, dtype=np.int32)))
    quad_active = np.concatenate((mesh.quad_active, np.ones(len(selected), dtype=bool)))
    result = MeshCore(
        mesh.node_coordinates,
        mesh.triangle_connectivity,
        quad_connectivity,
        node_ids=mesh.node_ids,
        triangle_ids=mesh.triangle_ids,
        quad_ids=quad_ids,
        owner_table=mesh.owner_table,
        node_owner_handles=mesh.node_owner_handles,
        triangle_owner_handles=mesh.triangle_owner_handles,
        quad_owner_handles=quad_owners,
        node_active=mesh.node_active,
        triangle_active=triangle_active,
        quad_active=quad_active,
    )
    pairs = np.asarray(
        [(int(mesh.triangle_ids[item.first]), int(mesh.triangle_ids[item.second])) for item in selected],
        dtype=np.int64,
    ).reshape((-1, 2))
    scores = np.asarray([item.score for item in selected], dtype=np.float64)
    pairs.setflags(write=False)
    created_ids.setflags(write=False)
    scores.setflags(write=False)
    if cancellation_check is not None:
        cancellation_check("triangle recombination complete")
    return RecombinationReport(
        result,
        pairs,
        created_ids,
        scores,
        len(candidates) + rejected,
        rejected,
        exchange_count,
        exchange_work,
        exchange_truncated,
    )


def recombine_triangles(
    mesh_or_points: MeshCore | Any,
    triangles: Any | None = None,
    *,
    return_report: bool = False,
    **kwargs: Any,
) -> MeshCore | RecombinationReport:
    report = recombine_triangles_with_report(mesh_or_points, triangles, **kwargs)
    return report if return_report else report.mesh


recombine_to_quads = recombine_triangles
pair_triangles = recombine_triangles_with_report

