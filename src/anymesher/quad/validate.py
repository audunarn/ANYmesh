"""Independent O(N+E) validation for PQ-M1 planar quad results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np

from ..errors import MeshError
from ..triangulation import orient2d
from .front import EPS, edge_key
from .seed import PlanarQuadSeed

__all__ = ["QuadValidationReport", "validate_planar_quad_result"]


@dataclass(frozen=True)
class QuadValidationReport:
    face: int
    node_count: int
    quad_count: int
    tri_count: int
    q4_count_fraction: float
    q4_area_fraction: float
    n_eq: float
    cell_area_sum: float
    reference_area: float
    area_ratio: float
    min_cell_area: float
    boundary_edge_count: int
    interior_edge_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "face": self.face,
            "node_count": self.node_count,
            "quad_count": self.quad_count,
            "tri_count": self.tri_count,
            "q4_count_fraction": self.q4_count_fraction,
            "q4_area_fraction": self.q4_area_fraction,
            "n_eq": self.n_eq,
            "cell_area_sum": self.cell_area_sum,
            "reference_area": self.reference_area,
            "area_ratio": self.area_ratio,
            "min_cell_area": self.min_cell_area,
            "boundary_edge_count": self.boundary_edge_count,
            "interior_edge_count": self.interior_edge_count,
        }


def _position(state, node: int) -> tuple[float, float]:
    try:
        pos = state.position(int(node))
    except Exception as exc:
        raise MeshError(f"cell references unknown node {node}") from exc
    if len(pos) != 2:
        raise MeshError(f"node {node} does not have two chart coordinates")
    x, y = float(pos[0]), float(pos[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        raise MeshError(f"node {node} has non-finite coordinates")
    return x, y


def _signed_area(points: Sequence[Tuple[float, float]]) -> float:
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )


def _strict_ccw_quad(points: Sequence[Tuple[float, float]]) -> bool:
    if len(points) != 4 or _signed_area(points) <= EPS:
        return False
    return all(
        orient2d(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) > EPS
        for i in range(4)
    )


def validate_planar_quad_result(
    state,
    *,
    face: int = 0,
    reference_area: float | None = None,
    seed: PlanarQuadSeed | None = None,
) -> QuadValidationReport:
    """Validate the resident final state without coordinate welding or O(N^2) scans."""
    if reference_area is not None and (
        not np.isfinite(float(reference_area)) or float(reference_area) <= 0.0
    ):
        raise MeshError("reference_area must be positive and finite")

    for node in state.nodes:
        _position(state, int(node))

    seen_bodies: set[tuple[int, ...]] = set()
    incidence: dict[tuple[int, int], int] = {}
    quad_count = tri_count = 0
    quad_area = tri_area = 0.0
    cell_areas: list[float] = []

    for cid in sorted(int(item) for item in state.cells):
        body = tuple(int(x) for x in state.cell(cid))
        kind = state.cell_kind(cid)
        expected = 4 if kind == "Q4" else 3 if kind == "T3" else 0
        if expected == 0:
            raise MeshError(f"cell {cid} has unsupported kind {kind!r}")
        if len(body) != expected or len(set(body)) != expected:
            raise MeshError(f"{kind} cell {cid} does not have {expected} distinct nodes")
        signature = tuple(sorted(body))
        if signature in seen_bodies:
            raise MeshError(f"duplicate active cell body at cell {cid}")
        seen_bodies.add(signature)
        points = [_position(state, node) for node in body]
        signed = _signed_area(points)
        if kind == "Q4":
            if not _strict_ccw_quad(points):
                raise MeshError(f"Q4 cell {cid} is not a simple strictly-positive quad")
            quad_count += 1
            quad_area += signed
        else:
            if signed <= EPS:
                raise MeshError(f"T3 cell {cid} is not positively oriented")
            tri_count += 1
            tri_area += signed
        cell_areas.append(signed)
        for i in range(len(body)):
            edge = edge_key(body[i], body[(i + 1) % len(body)])
            incidence[edge] = incidence.get(edge, 0) + 1

    if not cell_areas:
        raise MeshError("face has no active cells")
    for edge, count in incidence.items():
        if count not in (1, 2):
            raise MeshError(f"edge {edge} has nonmanifold active incidence {count}")

    expected_boundary: set[tuple[int, int]] | None = None
    if seed is not None:
        expected_boundary = {
            edge_key(int(raw[0]), int(raw[1]))
            for raw in seed.triangulation.boundary_segments
        }
        actual_boundary = {edge for edge, count in incidence.items() if count == 1}
        if actual_boundary != expected_boundary:
            missing = sorted(expected_boundary - actual_boundary)
            extra = sorted(actual_boundary - expected_boundary)
            raise MeshError(
                f"final boundary differs from canonical station boundary; missing={missing!r}, extra={extra!r}"
            )
        expected_interior_features = {
            edge_key(int(raw[0]), int(raw[1]))
            for raw in seed.triangulation.mandatory_segments
        }
        expected_nodes = {int(node) for node in seed.station_to_node.values()}
        expected_nodes.update(
            node for edge in expected_interior_features for node in edge
        )
        if set(state.protected_nodes) != expected_nodes:
            raise MeshError("protected seed-node ownership changed during PQ-M1")
        if set(state.protected_edges) != expected_boundary | expected_interior_features:
            raise MeshError("protected seed-edge ownership changed during PQ-M1")

    total = quad_area + tri_area
    if total <= EPS:
        raise MeshError("face has no positive cell area")
    if reference_area is None:
        ratio = float("nan")
        reference = float("nan")
    else:
        reference = float(reference_area)
        ratio = total / reference
        tol = max(1.0e-10, 5.0e-10 * max(1.0, reference))
        if abs(total - reference) > tol:
            raise MeshError(
                f"cell area sum {total:g} does not match reference area {reference:g}"
            )

    total_cells = quad_count + tri_count
    boundary_count = sum(1 for count in incidence.values() if count == 1)
    return QuadValidationReport(
        face=int(face),
        node_count=len(state.nodes),
        quad_count=quad_count,
        tri_count=tri_count,
        q4_count_fraction=float(quad_count / total_cells),
        q4_area_fraction=float(quad_area / total),
        n_eq=float(quad_count + 0.5 * tri_count),
        cell_area_sum=float(total),
        reference_area=reference,
        area_ratio=float(ratio),
        min_cell_area=float(min(cell_areas)),
        boundary_edge_count=boundary_count,
        interior_edge_count=len(incidence) - boundary_count,
    )
