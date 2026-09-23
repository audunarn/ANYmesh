"""Bounded TinyAD optimization of real resident Q4 mesh nodes."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable

from .patch_energy import PatchSpec, energy, is_valid_patch
from .quad_tinyad_worker import WorkerNotFoundQ5, find_worker, solve_q5_patch
from .state import QuadMeshState

_BUDGET_CAP = 8
_ENERGY_EPS = 1.0e-14
_IMPROVE_EPS = 1.0e-12


@dataclass(frozen=True)
class QuadOptimizerReport:
    status: str
    eligible_nodes: int
    attempts: int
    applied: int
    worker_calls: int
    objective_initial_sum: float
    objective_final_sum: float
    moved_node_ids: tuple[int, ...]
    max_displacement: float
    worker_statuses: tuple[str, ...]
    budget: int
    budget_cap: int = _BUDGET_CAP
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "eligible_nodes": self.eligible_nodes,
            "attempts": self.attempts,
            "applied": self.applied,
            "worker_calls": self.worker_calls,
            "objective_initial_sum": self.objective_initial_sum,
            "objective_final_sum": self.objective_final_sum,
            "moved_node_ids": list(self.moved_node_ids),
            "max_displacement": self.max_displacement,
            "worker_statuses": list(self.worker_statuses),
            "budget": self.budget,
            "budget_cap": self.budget_cap,
        }


def default_q5_worker_exe() -> Path:
    root = Path(__file__).resolve().parents[3]
    name = "quad_tinyad_optimizer.exe" if os.name == "nt" else "quad_tinyad_optimizer"
    return root / "third_party" / "quad" / "worker" / "out" / "tinyad" / name


def default_q5_worker_root() -> Path:
    return default_q5_worker_exe().parent


def _cancel(check: Callable[[str], None] | None, stage: str) -> None:
    if check is not None:
        check(stage)
def _report(status: str, *, eligible: int, budget: int, attempts: int = 0,
            applied: int = 0, calls: int = 0, initial: float = 0.0,
            final: float = 0.0, moved: tuple[int, ...] = (),
            max_disp: float = 0.0, statuses: tuple[str, ...] = ()) -> QuadOptimizerReport:
    return QuadOptimizerReport(
        status=status,
        eligible_nodes=eligible,
        attempts=attempts,
        applied=applied,
        worker_calls=calls,
        objective_initial_sum=float(initial),
        objective_final_sum=float(final),
        moved_node_ids=moved,
        max_displacement=float(max_disp),
        worker_statuses=statuses,
        budget=budget,
    )


def _incident_q4_cells(state: QuadMeshState, node: int) -> tuple[int, ...] | None:
    cells = state.cells_at(node)
    if not cells:
        return None
    if any(state.cell_kind(cid) != "Q4" for cid in cells):
        return None
    return tuple(cells)


def _eligible(state: QuadMeshState, node: int) -> bool:
    if state.is_protected_node(node):
        return False
    cells = _incident_q4_cells(state, node)
    if cells is None or len(cells) < 3:
        return False
    local_nodes = {nid for cid in cells for nid in state.cell(cid)}
    if any(nid != node and state.is_protected_node(nid) for nid in local_nodes):
        return False
    return True


def _patch_spec(state: QuadMeshState, center: int, h: float) -> PatchSpec:
    cells = _incident_q4_cells(state, center)
    if cells is None:
        raise ValueError("center is not an all-Q4 node")
    local_nodes = tuple(sorted({nid for cid in cells for nid in state.cell(cid)}))
    index = {nid: i for i, nid in enumerate(local_nodes)}
    quads = tuple(tuple(index[nid] for nid in state.cell(cid)) for cid in cells)
    nx = tuple(float(state.position(nid)[0]) for nid in local_nodes)
    ny = tuple(float(state.position(nid)[1]) for nid in local_nodes)
    return PatchSpec(
        h=float(h), ux=1.0, uy=0.0, nx=nx, ny=ny,
        free=(index[center],), quads=quads, max_iter=64,
    )


def _patch_energy(spec: PatchSpec) -> float:
    return float(energy(spec.nx, spec.ny, spec.quads, spec.h, spec.ux, spec.uy))


def _local_h(state: Any, node: int, target: float, size_field: Any, domain: Any) -> float:
    if size_field is None or domain is None:
        return target
    value = float(size_field.size_at(domain.lift(state.position(node)))[0])
    return value if math.isfinite(value) and value > 0.0 else target


def _resolve_worker(worker_exe: str | os.PathLike[str] | None,
                    worker_root: str | os.PathLike[str] | None) -> Path:
    if worker_exe is not None:
        return find_worker(worker_exe)
    if worker_root is not None:
        root = Path(worker_root)
        name = "quad_tinyad_optimizer.exe" if os.name == "nt" else "quad_tinyad_optimizer"
        return find_worker(root / name)
    return find_worker(None)
def optimize_quad_state(
    state: QuadMeshState,
    *,
    target_size: float,
    max_local_optimizations: int,
    worker_exe: str | os.PathLike[str] | None = None,
    worker_root: str | os.PathLike[str] | None = None,
    cancellation_check: Callable[[str], None] | None = None,
    size_field: Any = None,
    domain: Any = None,
    timeout: float = 30.0,
) -> QuadOptimizerReport:
    """Optimize a bounded set of real Q4 interior nodes in-place."""
    h = float(target_size)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("target_size must be positive and finite")
    budget = min(max(int(max_local_optimizations), 0), _BUDGET_CAP)
    if budget == 0:
        return _report("DISABLED", eligible=0, budget=0)

    candidates: list[tuple[float, int]] = []
    eligible_count = 0
    for node in sorted(state.nodes):
        if not _eligible(state, node):
            continue
        eligible_count += 1
        local_h = _local_h(state, node, h, size_field, domain)
        spec = _patch_spec(state, node, local_h)
        e0 = _patch_energy(spec)
        if math.isfinite(e0) and e0 > _ENERGY_EPS:
            candidates.append((e0, node))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if eligible_count == 0:
        return _report("NO_ELIGIBLE", eligible=0, budget=budget)
    if not candidates:
        return _report("NOIMPROVE", eligible=eligible_count, budget=budget)
    if os.environ.get("ANYMESH_Q5_DISABLE_WORKER") == "1":
        return _report("UNAVAILABLE_SKIPPED", eligible=eligible_count, budget=budget)
    try:
        worker = _resolve_worker(worker_exe, worker_root)
    except WorkerNotFoundQ5:
        return _report("UNAVAILABLE_SKIPPED", eligible=eligible_count, budget=budget)

    attempts = applied = calls = 0
    initial_sum = final_sum = 0.0
    moved: list[int] = []
    statuses: list[str] = []
    max_disp = 0.0

    for _score, node in candidates[:budget]:
        local_h = _local_h(state, node, h, size_field, domain)
        spec = _patch_spec(state, node, local_h)
        e0 = _patch_energy(spec)
        if e0 <= _ENERGY_EPS:
            continue
        attempts += 1
        _cancel(cancellation_check, "quad-first:q5-worker")
        report = solve_q5_patch(spec, worker=worker, timeout=float(timeout))
        calls += 1
        statuses.append(str(report.status))
        initial_sum += float(report.objective_initial)
        final_sum += float(report.objective_final)
        tol = max(_IMPROVE_EPS, abs(float(report.objective_initial)) * 1.0e-12)
        if report.status != "CONVERGED" or report.objective_final >= report.objective_initial - tol:
            continue
        if not report.free_final:
            continue
        new_pos = tuple(map(float, report.free_final[0]))
        old_pos = state.position(node)
        tx = state.transaction()
        tx.move_node(node, new_pos)
        candidate = tx.view
        check_spec = _patch_spec(candidate, node, local_h)
        if not is_valid_patch(check_spec.nx, check_spec.ny, check_spec.quads):
            tx.rollback()
            continue
        _cancel(cancellation_check, "quad-first:q5-commit")
        tx.commit()
        applied += 1
        moved.append(node)
        max_disp = max(max_disp, math.hypot(new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]))

    status = "APPLIED" if applied else "NOIMPROVE"
    return _report(
        status,
        eligible=eligible_count,
        budget=budget,
        attempts=attempts,
        applied=applied,
        calls=calls,
        initial=initial_sum,
        final=final_sum,
        moved=tuple(moved),
        max_disp=max_disp,
        statuses=tuple(statuses),
    )


__all__ = [
    "QuadOptimizerReport",
    "optimize_quad_state",
    "default_q5_worker_exe",
    "default_q5_worker_root",
]
