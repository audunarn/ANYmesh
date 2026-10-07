"""Static mesh-element BVH and isoparametric inverse interpolation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ._runtime_counters import (
    MESHER_BVH_BUILDS,
    MESHER_BVH_LOOKUPS,
    add_operation_count,
)
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


class _NormalizedElementFilter(frozenset):
    """PRIVATE trusted element-ID filter for repeated internal queries.

    Members are already ``int`` element IDs.  Only this exact type skips
    the public per-member ``int`` coercion in
    :meth:`MeshElementBVH.locate_all`; every other iterable, including
    plain frozensets, is coerced member by member.
    """


def normalized_element_filter(
    element_ids: Iterable[int],
) -> _NormalizedElementFilter:
    """Return a private trusted filter of coerced element IDs (internal)."""

    return _NormalizedElementFilter(int(value) for value in element_ids)


class MeshElementBVH:
    """A static AABB hierarchy over shell elements with a mutable active mask.

    The public constructor eagerly snapshots every selected shell's
    coordinates and builds the tree: later mesh mutations never change
    query results, and malformed coordinates fail at construction, exactly
    as before.  The private :meth:`_connectivity_only` constructor performs
    the same shell-input validation but defers the coordinate records and
    the tree to the first locating operation, for internal connectivity
    paths that may never locate a point.
    """

    def __init__(
        self,
        mesh: Mesh,
        *,
        element_ids: Iterable[int] | None = None,
        tolerance: float = 1.0e-9,
        leaf_size: int = 8,
    ) -> None:
        self._init_shell_state(mesh, element_ids, tolerance, leaf_size)
        self._ensure_built()

    @classmethod
    def _connectivity_only(
        cls,
        mesh: Mesh,
        *,
        element_ids: Iterable[int] | None = None,
        tolerance: float = 1.0e-9,
        leaf_size: int = 8,
    ) -> "MeshElementBVH":
        """PRIVATE deferred constructor for connectivity-only pipelines.

        Shell-input validation (duplicate shell IDs, unsupported
        connectivity lengths, missing nodes, malformed node coordinates)
        stays eager with the same rejections as the public constructor.
        The per-element AABB records and the tree are built lazily by
        the first locating operation, so a pipeline that never locates
        a point never pays for a tree.
        """

        self = cls.__new__(cls)
        self._init_shell_state(mesh, element_ids, tolerance, leaf_size)
        return self

    def _init_shell_state(
        self,
        mesh: Mesh,
        element_ids: Iterable[int] | None,
        tolerance: float,
        leaf_size: int,
    ) -> None:
        self.mesh = mesh
        self.tolerance = float(tolerance)
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise MeshError("BVH tolerance must be non-negative")
        if int(leaf_size) <= 0:
            raise MeshError("BVH leaf size must be positive")
        selected = None if element_ids is None else {int(value) for value in element_ids}
        overlap = set(mesh.quads).intersection(mesh.tris)
        if overlap:
            raise MeshError(f"shell element IDs occur as both quad and triangle: {sorted(overlap)}")
        shells: dict[int, tuple[ElementType, tuple[int, ...]]] = {}
        coordinates: dict[int, np.ndarray] = {}
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
                for node in node_ids:
                    if node not in mesh.nodes:
                        raise MeshError(
                            f"element {element_id} references missing node {node}"
                        )
                # Eager per-element coordinate construction keeps the
                # historical rejection of ragged or non-numeric node
                # coordinates on every construction path, without building
                # the records or the tree.
                coordinates[int(element_id)] = np.asarray(
                    [mesh.nodes[node] for node in node_ids], dtype=float
                )
                shells[int(element_id)] = (kind, node_ids)
        self._shells = shells
        self._shell_coordinates = coordinates
        self._records: dict[int, _Record] | None = None
        self._active = set(shells)
        self._root: _Node | None = None
        self._leaf_size = int(leaf_size)

    def _ensure_built(self) -> dict[int, _Record]:
        """Build the element records and the AABB tree once, on first need."""

        records = self._records
        if records is not None:
            return records
        records = {}
        for element_id, (kind, node_ids) in self._shells.items():
            coordinates = self._shell_coordinates[element_id]
            records[element_id] = _Record(
                element_id,
                kind,
                node_ids,
                coordinates,
                coordinates.min(axis=0),
                coordinates.max(axis=0),
            )
        self._records = records
        self._root = self._build(tuple(sorted(records)), self._leaf_size)
        add_operation_count(MESHER_BVH_BUILDS)
        return records

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
        positions = {identifier: position for position, identifier in enumerate(identifiers)}
        ordered = tuple(sorted(identifiers, key=lambda item: (centers[positions[item], axis], item)))
        middle = len(ordered) // 2
        return _Node(
            lower,
            upper,
            left=self._build(ordered[:middle], leaf_size),
            right=self._build(ordered[middle:], leaf_size),
        )

    @property
    def element_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._shells))

    @property
    def active_elements(self) -> frozenset[int]:
        return frozenset(self._active)

    def set_active(self, element_ids: Iterable[int], active: bool = True) -> None:
        identifiers = {int(value) for value in element_ids}
        unknown = identifiers - set(self._shells)
        if unknown:
            raise MeshError(f"BVH has no shell elements {sorted(unknown)}")
        if active:
            self._active.update(identifiers)
        else:
            self._active.difference_update(identifiers)

    def replace_active(self, element_ids: Iterable[int]) -> None:
        identifiers = {int(value) for value in element_ids}
        unknown = identifiers - set(self._shells)
        if unknown:
            raise MeshError(f"BVH has no shell elements {sorted(unknown)}")
        self._active = identifiers

    @staticmethod
    def _intersects(
        node: _Node, lower: np.ndarray, upper: np.ndarray
    ) -> bool:
        return bool(np.all(node.upper >= lower) and np.all(node.lower <= upper))

    def _traverse_bounds(
        self, made_lower: np.ndarray, made_upper: np.ndarray
    ) -> tuple[int, ...]:
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

    def query_bounds(self, lower: object, upper: object) -> tuple[int, ...]:
        made_lower = np.asarray(lower, dtype=float)
        made_upper = np.asarray(upper, dtype=float)
        if made_lower.shape != (3,) or made_upper.shape != (3,):
            raise MeshError("BVH bounds must be 3-vectors")
        self._ensure_built()
        add_operation_count(MESHER_BVH_LOOKUPS)
        return self._traverse_bounds(made_lower, made_upper)

    def _candidate_ids(self, point: object, tolerance: float | None) -> tuple[int, ...]:
        made = np.asarray(point, dtype=float)
        if made.shape != (3,) or not np.all(np.isfinite(made)):
            raise MeshError("BVH query point must be a finite 3-vector")
        pad = self.tolerance if tolerance is None else float(tolerance)
        return self._traverse_bounds(made - pad, made + pad)

    def candidates(self, point: object, tolerance: float | None = None) -> tuple[int, ...]:
        self._ensure_built()
        add_operation_count(MESHER_BVH_LOOKUPS)
        return self._candidate_ids(point, tolerance)

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
        if element_ids is None:
            allowed = None
        elif type(element_ids) is _NormalizedElementFilter:
            # Private trusted filters already contain int element IDs and
            # are reused by callers that query one target repeatedly.
            allowed = element_ids
        else:
            allowed = {int(value) for value in element_ids}
        records = self._ensure_built()
        add_operation_count(MESHER_BVH_LOOKUPS)
        candidates = self._candidate_ids(made_point, made_tolerance)
        hits: list[ElementHit] = []
        for element_id in candidates:
            if allowed is not None and element_id not in allowed:
                continue
            record = records[element_id]
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

