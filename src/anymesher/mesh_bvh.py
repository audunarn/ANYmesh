"""Static mesh-element BVH and isoparametric inverse interpolation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .errors import MeshError
from .mesh import Mesh

__all__ = [
    "ElementHit",
    "ElementType",
    "InverseInterpolation",
    "MeshElementBVH",
    "inverse_interpolate",
    "shape_functions",
    "shape_functions_q4",
    "shape_functions_q8",
    "shape_functions_t3",
    "shape_functions_t6",
]


class ElementType(StrEnum):
    Q4 = "Q4"
    Q8 = "Q8"
    T3 = "T3"
    T6 = "T6"


def _element_type(value: ElementType | str) -> ElementType:
    if isinstance(value, ElementType):
        return value
    try:
        return ElementType(str(value).upper())
    except ValueError as error:
        raise MeshError(f"unsupported shell element type {value!r}") from error


def shape_functions_q4(xi: float, eta: float) -> np.ndarray:
    return np.asarray(
        (
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ),
        dtype=float,
    )


def shape_functions_q8(xi: float, eta: float) -> np.ndarray:
    return np.asarray(
        (
            -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
            -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
            -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
            -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
            0.5 * (1.0 - xi * xi) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta * eta),
            0.5 * (1.0 - xi * xi) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta * eta),
        ),
        dtype=float,
    )


def shape_functions_t3(r: float, s: float) -> np.ndarray:
    return np.asarray((1.0 - r - s, r, s), dtype=float)


def shape_functions_t6(r: float, s: float) -> np.ndarray:
    a = 1.0 - r - s
    return np.asarray(
        (
            a * (2.0 * a - 1.0),
            r * (2.0 * r - 1.0),
            s * (2.0 * s - 1.0),
            4.0 * a * r,
            4.0 * r * s,
            4.0 * s * a,
        ),
        dtype=float,
    )


def shape_functions(
    element_type: ElementType | str, first: float, second: float
) -> np.ndarray:
    made = _element_type(element_type)
    if made is ElementType.Q4:
        return shape_functions_q4(first, second)
    if made is ElementType.Q8:
        return shape_functions_q8(first, second)
    if made is ElementType.T3:
        return shape_functions_t3(first, second)
    return shape_functions_t6(first, second)


@dataclass(frozen=True, slots=True)
class InverseInterpolation:
    element_type: ElementType
    natural_coordinates: tuple[float, float]
    weights: tuple[float, ...]
    point: tuple[float, float, float]
    residual: float
    inside: bool

    @property
    def natural(self) -> tuple[float, float]:
        return self.natural_coordinates

    @property
    def projected(self) -> np.ndarray:
        return np.asarray(self.point, dtype=float)


def _initial_coordinates(kind: ElementType, coordinates: np.ndarray, point: np.ndarray) -> np.ndarray:
    corners = coordinates[:4] if kind in (ElementType.Q4, ElementType.Q8) else coordinates[:3]
    if kind in (ElementType.T3, ElementType.T6):
        matrix = np.column_stack((corners[1] - corners[0], corners[2] - corners[0]))
        made, *_ = np.linalg.lstsq(matrix, point - corners[0], rcond=None)
        return np.asarray(made, dtype=float)
    center = corners.mean(axis=0)
    d_xi = 0.25 * (-corners[0] + corners[1] + corners[2] - corners[3])
    d_eta = 0.25 * (-corners[0] - corners[1] + corners[2] + corners[3])
    made, *_ = np.linalg.lstsq(
        np.column_stack((d_xi, d_eta)), point - center, rcond=None
    )
    return np.asarray(made, dtype=float)


def _shape_derivatives(kind: ElementType, natural: np.ndarray) -> np.ndarray:
    first, second = float(natural[0]), float(natural[1])
    step_first = np.sqrt(np.finfo(float).eps) * max(1.0, abs(first))
    step_second = np.sqrt(np.finfo(float).eps) * max(1.0, abs(second))
    first_column = (
        shape_functions(kind, first + step_first, second)
        - shape_functions(kind, first - step_first, second)
    ) / (2.0 * step_first)
    second_column = (
        shape_functions(kind, first, second + step_second)
        - shape_functions(kind, first, second - step_second)
    ) / (2.0 * step_second)
    return np.column_stack((first_column, second_column))


def inverse_interpolate(
    element_type: ElementType | str,
    coordinates: object,
    point: object,
    *,
    tolerance: float = 1.0e-9,
    max_iterations: int = 30,
    require_inside: bool = True,
) -> InverseInterpolation | None:
    """Invert a Q4/Q8/T3/T6 isoparametric map in three dimensions.

    For compatibility with point-first callers, the two array arguments are
    swapped automatically when their shapes unambiguously identify them.
    """

    kind = _element_type(element_type)
    made_coordinates = np.asarray(coordinates, dtype=float)
    made_point = np.asarray(point, dtype=float)
    if made_coordinates.shape == (3,) and made_point.ndim == 2:
        made_coordinates, made_point = made_point, made_coordinates
    expected = int(kind.value[1:])
    if made_coordinates.shape != (expected, 3):
        raise MeshError(
            f"{kind.value} inverse interpolation needs an ({expected}, 3) "
            "coordinate array"
        )
    if made_point.shape != (3,) or not np.all(np.isfinite(made_point)):
        raise MeshError("inverse interpolation point must be a finite 3-vector")
    if not np.all(np.isfinite(made_coordinates)):
        raise MeshError("element coordinates must be finite")
    made_tolerance = float(tolerance)
    if not np.isfinite(made_tolerance) or made_tolerance < 0.0:
        raise MeshError("inverse interpolation tolerance must be non-negative")
    if int(max_iterations) <= 0:
        raise MeshError("max_iterations must be positive")

    natural = _initial_coordinates(kind, made_coordinates, made_point)
    converged = False
    for _ in range(int(max_iterations)):
        weights = shape_functions(kind, float(natural[0]), float(natural[1]))
        current = weights @ made_coordinates
        derivatives = _shape_derivatives(kind, natural)
        jacobian = made_coordinates.T @ derivatives
        if np.linalg.matrix_rank(jacobian) < 2:
            return None
        delta, *_ = np.linalg.lstsq(jacobian, made_point - current, rcond=None)
        natural += delta
        if float(np.linalg.norm(delta)) <= 5.0e-13:
            converged = True
            break
    weights = shape_functions(kind, float(natural[0]), float(natural[1]))
    projected = weights @ made_coordinates
    residual = float(np.linalg.norm(projected - made_point))
    extent = max(float(np.max(np.ptp(made_coordinates, axis=0))), 1.0)
    natural_tolerance = max(1.0e-10, made_tolerance / extent)
    first, second = float(natural[0]), float(natural[1])
    if kind in (ElementType.Q4, ElementType.Q8):
        inside = (
            -1.0 - natural_tolerance <= first <= 1.0 + natural_tolerance
            and -1.0 - natural_tolerance <= second <= 1.0 + natural_tolerance
        )
    else:
        inside = (
            first >= -natural_tolerance
            and second >= -natural_tolerance
            and first + second <= 1.0 + natural_tolerance
        )
    if not converged and residual > max(made_tolerance, 1.0e-12 * extent):
        return None
    if residual > made_tolerance:
        return None
    if require_inside and not inside:
        return None
    return InverseInterpolation(
        kind,
        (first, second),
        tuple(float(value) for value in weights),
        tuple(float(value) for value in projected),
        residual,
        inside,
    )


@dataclass(frozen=True, slots=True)
class ElementHit:
    element_id: int
    element_type: ElementType
    node_ids: tuple[int, ...]
    natural_coordinates: tuple[float, float]
    weights: tuple[float, ...]
    point: tuple[float, float, float]
    residual: float

    @property
    def natural(self) -> tuple[float, float]:
        return self.natural_coordinates


@dataclass(frozen=True, slots=True)
class _Record:
    element_id: int
    element_type: ElementType
    node_ids: tuple[int, ...]
    coordinates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True, slots=True)
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    elements: tuple[int, ...] = ()
    left: "_Node | None" = None
    right: "_Node | None" = None


class MeshElementBVH:
    """A static AABB hierarchy over shell elements with a mutable active mask."""

    def __init__(
        self,
        mesh: Mesh,
        *,
        element_ids: Iterable[int] | None = None,
        tolerance: float = 1.0e-9,
        leaf_size: int = 8,
    ) -> None:
        self.mesh = mesh
        self.tolerance = float(tolerance)
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise MeshError("BVH tolerance must be non-negative")
        if int(leaf_size) <= 0:
            raise MeshError("BVH leaf size must be positive")
        selected = None if element_ids is None else {int(value) for value in element_ids}
        records: dict[int, _Record] = {}
        overlap = set(mesh.quads).intersection(mesh.tris)
        if overlap:
            raise MeshError(f"shell element IDs occur as both quad and triangle: {sorted(overlap)}")
        for source, linear, quadratic in (
            (mesh.quads, ElementType.Q4, ElementType.Q8),
            (mesh.tris, ElementType.T3, ElementType.T6),
        ):
            for element_id, raw_nodes in source.items():
                if selected is not None and int(element_id) not in selected:
                    continue
                node_ids = tuple(int(value) for value in raw_nodes)
                kind = linear if len(node_ids) == int(linear.value[1:]) else quadratic
                if len(node_ids) != int(kind.value[1:]):
                    raise MeshError(
                        f"element {element_id} has unsupported connectivity length "
                        f"{len(node_ids)}"
                    )
                try:
                    coordinates = np.asarray(
                        [mesh.nodes[node] for node in node_ids], dtype=float
                    )
                except KeyError as error:
                    raise MeshError(
                        f"element {element_id} references missing node {error.args[0]}"
                    ) from error
                records[int(element_id)] = _Record(
                    int(element_id),
                    kind,
                    node_ids,
                    coordinates,
                    coordinates.min(axis=0),
                    coordinates.max(axis=0),
                )
        self._records = records
        self._active = set(records)
        self._root = self._build(tuple(sorted(records)), int(leaf_size))

    def _build(self, identifiers: tuple[int, ...], leaf_size: int) -> _Node | None:
        if not identifiers:
            return None
        lower = np.min([self._records[item].lower for item in identifiers], axis=0)
        upper = np.max([self._records[item].upper for item in identifiers], axis=0)
        if len(identifiers) <= leaf_size:
            return _Node(lower, upper, identifiers)
        centers = np.asarray(
            [
                0.5 * (self._records[item].lower + self._records[item].upper)
                for item in identifiers
            ]
        )
        axis = int(np.argmax(np.ptp(centers, axis=0)))
        ordered = tuple(sorted(identifiers, key=lambda item: (centers[identifiers.index(item), axis], item)))
        middle = len(ordered) // 2
        return _Node(
            lower,
            upper,
            left=self._build(ordered[:middle], leaf_size),
            right=self._build(ordered[middle:], leaf_size),
        )

    @property
    def element_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._records))

    @property
    def active_elements(self) -> frozenset[int]:
        return frozenset(self._active)

    def set_active(self, element_ids: Iterable[int], active: bool = True) -> None:
        identifiers = {int(value) for value in element_ids}
        unknown = identifiers - set(self._records)
        if unknown:
            raise MeshError(f"BVH has no shell elements {sorted(unknown)}")
        if active:
            self._active.update(identifiers)
        else:
            self._active.difference_update(identifiers)

    def replace_active(self, element_ids: Iterable[int]) -> None:
        identifiers = {int(value) for value in element_ids}
        unknown = identifiers - set(self._records)
        if unknown:
            raise MeshError(f"BVH has no shell elements {sorted(unknown)}")
        self._active = identifiers

    @staticmethod
    def _intersects(
        node: _Node, lower: np.ndarray, upper: np.ndarray
    ) -> bool:
        return bool(np.all(node.upper >= lower) and np.all(node.lower <= upper))

    def query_bounds(self, lower: object, upper: object) -> tuple[int, ...]:
        made_lower = np.asarray(lower, dtype=float)
        made_upper = np.asarray(upper, dtype=float)
        if made_lower.shape != (3,) or made_upper.shape != (3,):
            raise MeshError("BVH bounds must be 3-vectors")
        found: list[int] = []

        def visit(node: _Node | None) -> None:
            if node is None or not self._intersects(node, made_lower, made_upper):
                return
            if node.elements:
                found.extend(item for item in node.elements if item in self._active)
                return
            visit(node.left)
            visit(node.right)

        visit(self._root)
        return tuple(sorted(found))

    def candidates(self, point: object, tolerance: float | None = None) -> tuple[int, ...]:
        made = np.asarray(point, dtype=float)
        if made.shape != (3,) or not np.all(np.isfinite(made)):
            raise MeshError("BVH query point must be a finite 3-vector")
        pad = self.tolerance if tolerance is None else float(tolerance)
        return self.query_bounds(made - pad, made + pad)

    query_point = candidates

    def locate_all(
        self,
        point: object,
        *,
        element_ids: Iterable[int] | None = None,
        tolerance: float | None = None,
    ) -> tuple[ElementHit, ...]:
        made_point = np.asarray(point, dtype=float)
        made_tolerance = self.tolerance if tolerance is None else float(tolerance)
        allowed = None if element_ids is None else {int(value) for value in element_ids}
        candidates = self.candidates(made_point, made_tolerance)
        hits: list[ElementHit] = []
        for element_id in candidates:
            if allowed is not None and element_id not in allowed:
                continue
            record = self._records[element_id]
            inverse = inverse_interpolate(
                record.element_type,
                record.coordinates,
                made_point,
                tolerance=made_tolerance,
            )
            if inverse is None:
                continue
            hits.append(
                ElementHit(
                    element_id,
                    record.element_type,
                    record.node_ids,
                    inverse.natural_coordinates,
                    inverse.weights,
                    inverse.point,
                    inverse.residual,
                )
            )
        hits.sort(key=lambda item: (item.residual, item.element_id))
        return tuple(hits)

    def locate(
        self,
        point: object,
        *,
        element_ids: Iterable[int] | None = None,
        tolerance: float | None = None,
    ) -> ElementHit | None:
        hits = self.locate_all(
            point, element_ids=element_ids, tolerance=tolerance
        )
        return None if not hits else hits[0]

