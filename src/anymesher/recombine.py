"""Quality-ranked, deterministic T3-to-Q4 recombination."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Sequence

import numpy as np

from .core import MeshCore, corner_edges
from .errors import MeshError
from .quality_v2 import quad_candidate_quality

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
) -> RecombinationReport:
    """Pair adjacent triangles greedily from best quality to worst.

    A candidate which misses any hard threshold is left as two triangles.  The
    deterministic ordering uses quality first, then stable triangle/node IDs.
    Paired triangles remain in the dense table with inactive flags, preserving
    their IDs for diagnostics and incremental clients.
    """

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
    incidence = _edge_incidence(mesh)
    candidates: list[_Candidate] = []
    rejected = 0
    for edge, attached in sorted(incidence.items()):
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
    return RecombinationReport(result, pairs, created_ids, scores, len(candidates) + rejected, rejected)


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

