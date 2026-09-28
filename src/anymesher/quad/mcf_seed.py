"""Bounded geometry-derived Q4 MCF pairing on a real planar seed."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import os
from typing import Any, Callable

from .count_mcf import CountInfeasible
from .count_model import CountInstance
from .front import FrontRejected, body_edges, make_quad
from .quad_mcf_worker import WorkerNotFound, find_worker, solve_count_instance
from .state import QuadMeshState

_MAX_COMPONENT_CELLS = 12
_MAX_COMPONENT_ARCS = 12
_MAX_COMPONENTS = 4
_COST_SCALE = 10_000


@dataclass(frozen=True)
class Q4MCFReport:
    status: str
    candidate_components: int
    eligible_components: int
    solved_components: int
    worker_calls: int
    applied_pairs: int
    initial_t3: int
    final_t3: int
    initial_q4: int
    final_q4: int
    selected_pairs: tuple[tuple[int, int], ...]
    total_cost: int
    skipped_large: int = 0
    skipped_unbalanced: int = 0
    skipped_nonbipartite: int = 0
    skipped_infeasible: int = 0
    skipped_component_cap: int = 0
    component_sizes: tuple[int, ...] = ()
    arc_counts: tuple[int, ...] = ()
    added_q4_ids: tuple[int, ...] = ()
    generation_before: int = 0
    generation_after: int = 0

    component_bounds: tuple[int, int, int] = (
        _MAX_COMPONENTS, _MAX_COMPONENT_CELLS, _MAX_COMPONENT_ARCS
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "candidate_components": self.candidate_components,
            "eligible_components": self.eligible_components,
            "solved_components": self.solved_components,
            "worker_calls": self.worker_calls,
            "applied_pairs": self.applied_pairs,
            "initial_t3": self.initial_t3,
            "final_t3": self.final_t3,
            "initial_q4": self.initial_q4,
            "final_q4": self.final_q4,
            "selected_pairs": [list(pair) for pair in self.selected_pairs],
            "total_cost": self.total_cost,
            "skipped_large": self.skipped_large,
            "skipped_unbalanced": self.skipped_unbalanced,
            "skipped_nonbipartite": self.skipped_nonbipartite,
            "skipped_infeasible": self.skipped_infeasible,
            "skipped_component_cap": self.skipped_component_cap,
            "component_sizes": list(self.component_sizes),
            "arc_counts": list(self.arc_counts),
            "added_q4_ids": list(self.added_q4_ids),
            "generation_before": self.generation_before,
            "generation_after": self.generation_after,
            "component_bounds": list(self.component_bounds),
        }


def _count_kind(state: QuadMeshState, kind: str) -> int:
    return state.count_kind(kind)


def _cancel(check: Callable[[str], None] | None, stage: str) -> None:
    if check is not None:
        check(stage)


def _pair_quad(
    state: QuadMeshState, a: int, b: int
) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    ba = state.cell(a)
    bb = state.cell(b)
    shared = set(body_edges(ba)) & set(body_edges(bb))
    if len(shared) != 1:
        return None
    shared_edge = next(iter(shared))
    if state.is_protected_edge(shared_edge):
        return None
    boundary = (set(body_edges(ba)) ^ set(body_edges(bb)))
    if len(boundary) != 4:
        return None
    adjacent: dict[int, list[int]] = defaultdict(list)
    for u, v in boundary:
        adjacent[u].append(v)
        adjacent[v].append(u)
    if len(adjacent) != 4 or any(len(nbrs) != 2 for nbrs in adjacent.values()):
        return None
    start = min(adjacent)
    walk = [start]
    previous: int | None = None
    current = start
    for _ in range(3):
        choices = sorted(n for n in adjacent[current] if n != previous)
        if not choices:
            return None
        nxt = choices[0]
        walk.append(nxt)
        previous, current = current, nxt
    if len(set(walk)) != 4 or start not in adjacent[walk[-1]]:
        return None
    try:
        quad = make_quad(state, tuple(walk))
    except FrontRejected:
        return None
    return shared_edge, quad


def _pair_graph(state: QuadMeshState):
    edge_to_t3: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid in sorted(int(c) for c in state.cells):
        if state.cell_kind(cid) != "T3":
            continue
        for edge in body_edges(state.cell(cid)):
            edge_to_t3[edge].append(cid)

    adjacency: dict[int, set[int]] = defaultdict(set)
    pair_data: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, ...]]] = {}
    for _edge, incident in sorted(edge_to_t3.items()):
        if len(incident) != 2:
            continue
        a, b = sorted(incident)
        paired = _pair_quad(state, a, b)
        if paired is None:
            continue
        shared, quad = paired
        adjacency[a].add(b)
        adjacency[b].add(a)
        pair_data[(a, b)] = (shared, quad)
    return adjacency, pair_data


def _components(adjacency: dict[int, set[int]]) -> list[tuple[int, ...]]:
    seen: set[int] = set()
    result: list[tuple[int, ...]] = []
    for root in sorted(adjacency):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        nodes: list[int] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for other in sorted(adjacency[node], reverse=True):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(tuple(sorted(nodes)))
    return result


def _bipartition(
    component: tuple[int, ...], adjacency: dict[int, set[int]]
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    color = {component[0]: 0}
    queue = deque([component[0]])
    while queue:
        node = queue.popleft()
        for other in sorted(adjacency[node]):
            if other not in color:
                color[other] = 1 - color[node]
                queue.append(other)
            elif color[other] == color[node]:
                return None
    left = tuple(sorted(node for node, value in color.items() if value == 0))
    right = tuple(sorted(node for node, value in color.items() if value == 1))
    return left, right


def _local_h(
    state: QuadMeshState,
    quad: tuple[int, ...],
    target: float,
    size_field: Any,
    domain: Any,
) -> float:
    if size_field is None or domain is None:
        return target
    x = sum(state.position(node)[0] for node in quad) / 4.0
    y = sum(state.position(node)[1] for node in quad) / 4.0
    value = float(size_field.size_at(domain.lift((x, y)))[0])
    return value if math.isfinite(value) and value > 0.0 else target


def _quad_cost(
    state: QuadMeshState,
    quad: tuple[int, ...],
    target: float,
    size_field: Any,
    domain: Any,
) -> int:
    desired = _local_h(state, quad, target, size_field, domain)
    points = [state.position(node) for node in quad]
    lengths = [
        math.dist(points[i], points[(i + 1) % 4]) for i in range(4)
    ]
    penalty = sum(((length / desired) - 1.0) ** 2 for length in lengths)
    return max(0, int(round(_COST_SCALE * penalty)))


def _classify_components(adjacency: dict[int, set[int]]):
    all_components = _components(adjacency)
    component_sizes = tuple(len(component) for component in all_components)
    arc_counts = tuple(
        sum(len(adjacency[node]) for node in component) // 2
        for component in all_components
    )
    eligible_all = []
    skipped_large = 0
    skipped_unbalanced = 0
    skipped_nonbipartite = 0
    for component, arcs in zip(all_components, arc_counts):
        if len(component) > _MAX_COMPONENT_CELLS or arcs > _MAX_COMPONENT_ARCS:
            skipped_large += 1
            continue
        partition = _bipartition(component, adjacency)
        if partition is None:
            skipped_nonbipartite += 1
            continue
        left, right = partition
        if not left or len(left) != len(right):
            skipped_unbalanced += 1
            continue
        eligible_all.append((component, left, right, arcs))
    eligible_all.sort(key=lambda item: (min(item[0]), len(item[0]), item[3]))
    skipped_component_cap = max(0, len(eligible_all) - _MAX_COMPONENTS)
    eligible = eligible_all[:_MAX_COMPONENTS]
    return (
        all_components, eligible, skipped_large, skipped_unbalanced,
        skipped_nonbipartite, skipped_component_cap, component_sizes, arc_counts,
    )


def _eligible_components(adjacency: dict[int, set[int]]):
    all_components, eligible, *_rest = _classify_components(adjacency)
    return all_components, eligible


def _instance_for_component(
    state: QuadMeshState,
    left: tuple[int, ...],
    right: tuple[int, ...],
    pair_data: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, ...]]],
    target_size: float,
    size_field: Any,
    domain: Any,
) -> CountInstance:
    cost: list[list[int]] = []
    blocked: list[tuple[int, int]] = []
    for i, a in enumerate(left):
        row: list[int] = []
        for j, b in enumerate(right):
            data = pair_data.get(tuple(sorted((a, b))))
            if data is None:
                row.append(0)
                blocked.append((i, j))
            else:
                row.append(_quad_cost(state, data[1], target_size, size_field, domain))
        cost.append(row)
    return CountInstance.from_arrays([1] * len(left), [1] * len(right), cost, blocked)


def _make_report(
    status: str,
    *,
    candidates: int,
    eligible: int,
    solved: int,
    calls: int,
    pairs: tuple[tuple[int, int], ...],
    initial_t3: int,
    final_t3: int,
    initial_q4: int,
    final_q4: int,
    total_cost: int,
    skipped_large: int = 0,
    skipped_unbalanced: int = 0,
    skipped_nonbipartite: int = 0,
    skipped_infeasible: int = 0,
    skipped_component_cap: int = 0,
    component_sizes: tuple[int, ...] = (),
    arc_counts: tuple[int, ...] = (),
    added_q4_ids: tuple[int, ...] = (),
    generation_before: int = 0,
    generation_after: int | None = None,
) -> Q4MCFReport:
    if generation_after is None:
        generation_after = generation_before
    return Q4MCFReport(
        status=status,
        candidate_components=candidates,
        eligible_components=eligible,
        solved_components=solved,
        worker_calls=calls,
        applied_pairs=len(pairs),
        initial_t3=initial_t3,
        final_t3=final_t3,
        initial_q4=initial_q4,
        final_q4=final_q4,
        selected_pairs=pairs,
        total_cost=int(total_cost),
        skipped_large=int(skipped_large),
        skipped_unbalanced=int(skipped_unbalanced),
        skipped_nonbipartite=int(skipped_nonbipartite),
        skipped_infeasible=int(skipped_infeasible),
        skipped_component_cap=int(skipped_component_cap),
        component_sizes=tuple(int(value) for value in component_sizes),
        arc_counts=tuple(int(value) for value in arc_counts),
        added_q4_ids=tuple(int(cid) for cid in added_q4_ids),
        generation_before=int(generation_before),
        generation_after=int(generation_after),
    )


def optimize_q4_seed_mcf(
    state: QuadMeshState,
    *,
    target_size: float,
    worker: str | os.PathLike[str] | None = None,
    cancellation_check: Callable[[str], None] | None = None,
    size_field: Any = None,
    domain: Any = None,
    timeout: float = 30.0,
) -> Q4MCFReport:
    """Pair bounded real T3 seed corridors through the qualified Q4 MCF worker."""
    h = float(target_size)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("target_size must be positive and finite")

    generation_before = int(state.generation)
    initial_t3 = _count_kind(state, "T3")
    initial_q4 = _count_kind(state, "Q4")
    adjacency, pair_data = _pair_graph(state)
    (
        components,
        eligible,
        skipped_large,
        skipped_unbalanced,
        skipped_nonbipartite,
        skipped_component_cap,
        component_sizes,
        arc_counts,
    ) = _classify_components(adjacency)
    common = dict(
        candidates=len(components), eligible=len(eligible), solved=0, calls=0,
        pairs=(), initial_t3=initial_t3, final_t3=initial_t3,
        initial_q4=initial_q4, final_q4=initial_q4, total_cost=0,
        skipped_large=skipped_large,
        skipped_unbalanced=skipped_unbalanced,
        skipped_nonbipartite=skipped_nonbipartite,
        skipped_infeasible=0,
        skipped_component_cap=skipped_component_cap,
        component_sizes=component_sizes,
        arc_counts=arc_counts,
        added_q4_ids=(),
        generation_before=generation_before,
        generation_after=generation_before,
    )
    if not eligible:
        return _make_report("NO_ELIGIBLE", **common)
    if os.environ.get("ANYMESH_Q4_DISABLE_WORKER") == "1":
        return _make_report("UNAVAILABLE_SKIPPED", **common)
    try:
        resolved_worker = find_worker(worker)
    except WorkerNotFound:
        return _make_report("UNAVAILABLE_SKIPPED", **common)

    selected: list[tuple[int, int]] = []
    solved_components = 0
    worker_calls = 0
    total_cost = 0
    skipped_infeasible = 0
    for _component, left, right, _arcs in eligible:
        instance = _instance_for_component(
            state, left, right, pair_data, h, size_field, domain
        )
        _cancel(cancellation_check, "quad-first:q4-mcf-worker")
        worker_calls += 1
        try:
            solution = solve_count_instance(
                instance, worker=resolved_worker, timeout=float(timeout)
            )
        except CountInfeasible:
            skipped_infeasible += 1
            continue
        solved_components += 1
        total_cost += int(solution.total_cost)
        for (i, j), flow in sorted(solution.flows_by_index(instance).items()):
            if int(flow) != 1:
                continue
            pair = tuple(sorted((int(left[i]), int(right[j]))))
            selected.append(pair)

    selected = sorted(set(selected))
    selected_tuple = tuple(selected)
    if not selected:
        common.update(
            solved=solved_components,
            calls=worker_calls,
            total_cost=total_cost,
            skipped_infeasible=skipped_infeasible,
        )
        return _make_report("NO_MATCH", **common)

    touched: set[tuple[int, int]] = set()
    quads: list[tuple[int, int, int, int]] = []
    for a, b in selected:
        data = pair_data.get((a, b))
        if data is None:
            raise RuntimeError(f"selected Q4 MCF pair {(a, b)} is no longer resident")
        quad = tuple(int(node) for node in data[1])
        touched.update(body_edges(state.cell(a)))
        touched.update(body_edges(state.cell(b)))
        touched.update(body_edges(quad))
        quads.append(quad)

    added_q4_ids: list[int] = []
    with state.transaction() as tx:
        for (a, b), quad in zip(selected, quads):
            tx.remove_cell(a)
            tx.remove_cell(b)
            added_q4_ids.append(int(tx.allocate_cell(quad, "Q4")))
        for edge in sorted(touched):
            is_front_now = sum(
                1
                for cid in tx.view.edge_cells(edge)
                if tx.view.cell_kind(cid) == "T3"
            ) == 1
            was_front = state.is_front_edge(edge)
            if is_front_now and not was_front:
                tx.add_front_edge(edge[0], edge[1])
            elif was_front and not is_front_now:
                tx.remove_front_edge(edge[0], edge[1])
        _cancel(cancellation_check, "quad-first:q4-mcf-commit")
        tx.commit()

    final_t3 = _count_kind(state, "T3")
    final_q4 = _count_kind(state, "Q4")
    return _make_report(
        "APPLIED",
        candidates=len(components),
        eligible=len(eligible),
        solved=solved_components,
        calls=worker_calls,
        pairs=selected_tuple,
        initial_t3=initial_t3,
        final_t3=final_t3,
        initial_q4=initial_q4,
        final_q4=final_q4,
        total_cost=total_cost,
        skipped_large=skipped_large,
        skipped_unbalanced=skipped_unbalanced,
        skipped_nonbipartite=skipped_nonbipartite,
        skipped_infeasible=skipped_infeasible,
        skipped_component_cap=skipped_component_cap,
        component_sizes=component_sizes,
        arc_counts=arc_counts,
        added_q4_ids=tuple(added_q4_ids),
        generation_before=generation_before,
        generation_after=state.generation,
    )


__all__ = ["Q4MCFReport", "optimize_q4_seed_mcf"]
