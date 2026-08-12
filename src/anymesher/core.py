"""Dense, stable mesh storage used by the native hybrid mesher.

The historical :class:`anymesher.mesh.Mesh` is deliberately friendly to code
which builds a mesh a node at a time.  This container serves the other end of
the pipeline: fixed-width NumPy arrays, integer handles instead of object
arrays, and topology which is built once and then read many times.

Connectivity contains *row indices* into ``node_coordinates``.  Public node
and element IDs live in separate arrays and therefore survive compaction,
recombination, and changes in storage order.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .errors import MeshError

__all__ = ["CSRAdjacency", "MeshCompatibilityView", "MeshCore", "corner_edges"]


def _freeze(value: Any, dtype: np.dtype[Any] | type, *, ndim: int | None = None) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise MeshError(f"expected a {ndim}-dimensional array, got shape {array.shape}")
    array.setflags(write=False)
    return array


def _connectivity(value: Any | None, widths: tuple[int, ...], name: str) -> np.ndarray:
    if value is None:
        return _freeze(np.empty((0, widths[0]), dtype=np.int64), np.int64, ndim=2)
    array = np.asarray(value, dtype=np.int64)
    if array.ndim == 1 and array.size == 0:
        array = np.empty((0, widths[0]), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] not in widths:
        choices = " or ".join(str(width) for width in widths)
        raise MeshError(f"{name} connectivity must have {choices} columns")
    return _freeze(array, np.int64, ndim=2)


def _ids(value: Any | None, count: int, start: int, name: str) -> np.ndarray:
    array = np.arange(start, start + count, dtype=np.int64) if value is None else np.asarray(value, dtype=np.int64)
    if array.shape != (count,):
        raise MeshError(f"{name} IDs must have shape ({count},), got {array.shape}")
    if np.unique(array).size != count:
        raise MeshError(f"{name} IDs must be unique")
    return _freeze(array, np.int64, ndim=1)


def _activity(value: Any | None, count: int, name: str) -> np.ndarray:
    array = np.ones(count, dtype=bool) if value is None else np.asarray(value, dtype=bool)
    if array.shape != (count,):
        raise MeshError(f"{name} activity must have shape ({count},), got {array.shape}")
    return _freeze(array, bool, ndim=1)


def _owners_equal(first: Any, second: Any) -> bool:
    if first is second:
        return True
    try:
        equal = first == second
        return bool(equal) if np.ndim(equal) == 0 else False
    except Exception:
        return False


def _owner_handles(
    handles: Any | None,
    owners: Sequence[Any] | None,
    count: int,
    table: list[Any],
    name: str,
) -> np.ndarray:
    if handles is not None and owners is not None:
        raise MeshError(f"provide either {name}_owner_handles or {name}_owners, not both")
    if owners is not None:
        if len(owners) != count:
            raise MeshError(f"{name}_owners must contain {count} entries")
        encoded = np.empty(count, dtype=np.int32)
        for row, owner in enumerate(owners):
            if owner is None:
                encoded[row] = -1
                continue
            handle = next(
                (index for index, candidate in enumerate(table) if _owners_equal(owner, candidate)),
                -1,
            )
            if handle < 0:
                table.append(owner)
                handle = len(table) - 1
            encoded[row] = handle
        return _freeze(encoded, np.int32, ndim=1)
    encoded = np.full(count, -1, dtype=np.int32) if handles is None else np.asarray(handles, dtype=np.int32)
    if encoded.shape != (count,):
        raise MeshError(f"{name}_owner_handles must have shape ({count},), got {encoded.shape}")
    if np.any(encoded < -1) or (encoded.size and np.any(encoded >= len(table))):
        raise MeshError(f"{name} owner handle is outside the owner table")
    return _freeze(encoded, np.int32, ndim=1)


def corner_edges(connectivity: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return undirected corner edges for T3/T6 or Q4/Q8 connectivity."""

    width = len(connectivity)
    if width in (3, 6):
        corners = connectivity[:3]
    elif width in (4, 8):
        corners = connectivity[:4]
    else:
        raise MeshError(f"unsupported shell connectivity width {width}")
    return tuple(
        (min(int(corners[index]), int(corners[(index + 1) % len(corners)])),
         max(int(corners[index]), int(corners[(index + 1) % len(corners)])))
        for index in range(len(corners))
    )


def _boundary_sequence(connectivity: np.ndarray) -> tuple[int, ...]:
    values = tuple(int(value) for value in connectivity)
    if len(values) == 3:
        return values
    if len(values) == 4:
        return values
    if len(values) == 6:
        return (values[0], values[3], values[1], values[4], values[2], values[5])
    if len(values) == 8:
        return (values[0], values[4], values[1], values[5], values[2], values[6], values[3], values[7])
    raise MeshError(f"unsupported shell connectivity width {len(values)}")


@dataclass(frozen=True)
class CSRAdjacency:
    """A small dependency-free CSR relation.

    Rows with no live relation are represented by equal consecutive pointers.
    ``indices`` are sorted within each row, making the result deterministic.
    """

    indptr: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        indptr = _freeze(self.indptr, np.int64, ndim=1)
        indices = _freeze(self.indices, np.int64, ndim=1)
        if indptr.size == 0 or int(indptr[0]) != 0 or int(indptr[-1]) != indices.size:
            raise MeshError("invalid CSR pointers")
        if np.any(np.diff(indptr) < 0):
            raise MeshError("CSR pointers must be nondecreasing")
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)

    @property
    def row_count(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def shape(self) -> tuple[int, int]:
        columns = int(np.max(self.indices)) + 1 if self.indices.size else 0
        return self.row_count, columns

    def row(self, index: int) -> np.ndarray:
        if index < 0:
            index += self.row_count
        if index < 0 or index >= self.row_count:
            raise IndexError(index)
        return self.indices[int(self.indptr[index]):int(self.indptr[index + 1])]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.row(index)


def _csr(rows: Sequence[set[int]]) -> CSRAdjacency:
    counts = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    indptr = np.empty(len(rows) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    cursor = 0
    for row in rows:
        values = sorted(row)
        indices[cursor:cursor + len(values)] = values
        cursor += len(values)
    return CSRAdjacency(indptr, indices)


class _CoreMapping(Mapping[int, Any]):
    __slots__ = ("_core", "_kind")

    def __init__(self, core: "MeshCore", kind: str) -> None:
        self._core = core
        self._kind = kind

    def _ids_and_activity(self) -> tuple[np.ndarray, np.ndarray]:
        if self._kind == "node":
            return self._core.node_ids, self._core.node_active
        if self._kind == "triangle":
            return self._core.triangle_ids, self._core.triangle_active
        return self._core.quad_ids, self._core.quad_active

    def __len__(self) -> int:
        return int(np.count_nonzero(self._ids_and_activity()[1]))

    def __iter__(self) -> Iterator[int]:
        ids, active = self._ids_and_activity()
        return (int(value) for value in ids[active])

    def __getitem__(self, stable_id: int) -> Any:
        row = self._core._row_for_id(self._kind, int(stable_id))
        _, active = self._ids_and_activity()
        if not bool(active[row]):
            raise KeyError(stable_id)
        if self._kind == "node":
            return self._core.node_coordinates[row]
        connectivity = (
            self._core.triangle_connectivity[row]
            if self._kind == "triangle"
            else self._core.quad_connectivity[row]
        )
        return tuple(int(self._core.node_ids[index]) for index in connectivity)


class _ShellMapping(Mapping[int, tuple[int, ...]]):
    __slots__ = ("_core",)

    def __init__(self, core: "MeshCore") -> None:
        self._core = core

    def __len__(self) -> int:
        return self._core.active_triangle_count + self._core.active_quad_count

    def __iter__(self) -> Iterator[int]:
        return iter((*self._core.tris, *self._core.quads))

    def __getitem__(self, stable_id: int) -> tuple[int, ...]:
        try:
            return self._core.tris[stable_id]
        except KeyError:
            return self._core.quads[stable_id]


class MeshCompatibilityView:
    """Lazy dictionary-shaped access compatible with the neutral ``Mesh`` API."""

    __slots__ = ("_core",)

    def __init__(self, core: "MeshCore") -> None:
        self._core = core

    @property
    def nodes(self) -> Mapping[int, np.ndarray]:
        return self._core.nodes

    @property
    def tris(self) -> Mapping[int, tuple[int, ...]]:
        return self._core.tris

    @property
    def quads(self) -> Mapping[int, tuple[int, ...]]:
        return self._core.quads

    @property
    def shells(self) -> Mapping[int, tuple[int, ...]]:
        return self._core.shells

    @property
    def order(self) -> str:
        return self._core.order

    @property
    def is_quadratic(self) -> bool:
        return self._core.is_quadratic

    @property
    def num_nodes(self) -> int:
        return self._core.active_node_count

    @property
    def num_elements(self) -> int:
        return self._core.active_element_count

    def corners_of(self, element_id: int) -> tuple[int, ...]:
        return self._core.corners_of(element_id)

    def node_positions(self) -> np.ndarray:
        return self._core.node_positions()


class MeshCore:
    """Compact shell mesh arrays with stable IDs and lazy derived topology."""

    def __init__(
        self,
        node_coordinates: Any | None = None,
        triangle_connectivity: Any | None = None,
        quad_connectivity: Any | None = None,
        *,
        coordinates: Any | None = None,
        triangles: Any | None = None,
        quadrilaterals: Any | None = None,
        node_ids: Any | None = None,
        triangle_ids: Any | None = None,
        quad_ids: Any | None = None,
        owner_table: Sequence[Any] = (),
        node_owner_handles: Any | None = None,
        triangle_owner_handles: Any | None = None,
        quad_owner_handles: Any | None = None,
        node_owners: Sequence[Any] | None = None,
        triangle_owners: Sequence[Any] | None = None,
        quad_owners: Sequence[Any] | None = None,
        node_active: Any | None = None,
        triangle_active: Any | None = None,
        quad_active: Any | None = None,
    ) -> None:
        if node_coordinates is not None and coordinates is not None:
            raise MeshError("provide node_coordinates or coordinates, not both")
        if triangle_connectivity is not None and triangles is not None:
            raise MeshError("provide triangle_connectivity or triangles, not both")
        if quad_connectivity is not None and quadrilaterals is not None:
            raise MeshError("provide quad_connectivity or quadrilaterals, not both")
        raw_coordinates = coordinates if node_coordinates is None else node_coordinates
        if raw_coordinates is None:
            raw_coordinates = np.empty((0, 3), dtype=float)
        coords = np.asarray(raw_coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] not in (2, 3):
            raise MeshError("node coordinates must have shape (n, 2) or (n, 3)")
        if coords.shape[1] == 2:
            coords = np.column_stack((coords, np.zeros(coords.shape[0], dtype=float)))
        if not np.all(np.isfinite(coords)):
            raise MeshError("node coordinates must be finite")
        self.node_coordinates = _freeze(coords, np.float64, ndim=2)
        self.triangle_connectivity = _connectivity(
            triangles if triangle_connectivity is None else triangle_connectivity,
            (3, 6),
            "triangle",
        )
        self.quad_connectivity = _connectivity(
            quadrilaterals if quad_connectivity is None else quad_connectivity,
            (4, 8),
            "quadrilateral",
        )

        node_count = self.node_coordinates.shape[0]
        for name, connectivity in (
            ("triangle", self.triangle_connectivity),
            ("quadrilateral", self.quad_connectivity),
        ):
            if connectivity.size and (int(np.min(connectivity)) < 0 or int(np.max(connectivity)) >= node_count):
                raise MeshError(f"{name} connectivity contains an invalid node row")

        triangle_count = self.triangle_connectivity.shape[0]
        quad_count = self.quad_connectivity.shape[0]
        self.node_ids = _ids(node_ids, node_count, 1, "node")
        self.triangle_ids = _ids(triangle_ids, triangle_count, 1, "triangle")
        self.quad_ids = _ids(quad_ids, quad_count, triangle_count + 1, "quadrilateral")
        if np.unique(np.concatenate((self.triangle_ids, self.quad_ids))).size != triangle_count + quad_count:
            raise MeshError("triangle and quadrilateral IDs must share one unique element namespace")

        table = list(owner_table)
        self.node_owner_handles = _owner_handles(
            node_owner_handles, node_owners, node_count, table, "node"
        )
        self.triangle_owner_handles = _owner_handles(
            triangle_owner_handles, triangle_owners, triangle_count, table, "triangle"
        )
        self.quad_owner_handles = _owner_handles(
            quad_owner_handles, quad_owners, quad_count, table, "quad"
        )
        self.owner_table = tuple(table)

        self.node_active = _activity(node_active, node_count, "node")
        self.triangle_active = _activity(triangle_active, triangle_count, "triangle")
        self.quad_active = _activity(quad_active, quad_count, "quadrilateral")
        for active, connectivity, name in (
            (self.triangle_active, self.triangle_connectivity, "triangle"),
            (self.quad_active, self.quad_connectivity, "quadrilateral"),
        ):
            if np.any(active):
                used = connectivity[active].ravel()
                if used.size and not np.all(self.node_active[used]):
                    raise MeshError(f"an active {name} references an inactive node")

        active_widths = set()
        if np.any(self.triangle_active):
            active_widths.add(self.triangle_connectivity.shape[1] // 3)
        if np.any(self.quad_active):
            active_widths.add(self.quad_connectivity.shape[1] // 4)
        if len(active_widths) > 1:
            raise MeshError("active triangles and quadrilaterals must use the same polynomial order")

        self._cache: dict[str, Any] = {}

    @property
    def coordinates(self) -> np.ndarray:
        return self.node_coordinates

    @property
    def points(self) -> np.ndarray:
        return self.node_coordinates

    @property
    def triangles(self) -> np.ndarray:
        return self.triangle_connectivity

    @property
    def quadrilaterals(self) -> np.ndarray:
        return self.quad_connectivity

    @property
    def num_nodes(self) -> int:
        return int(self.node_coordinates.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.triangle_connectivity.shape[0])

    @property
    def num_quads(self) -> int:
        return int(self.quad_connectivity.shape[0])

    @property
    def num_elements(self) -> int:
        return self.num_triangles + self.num_quads

    @property
    def active_node_count(self) -> int:
        return int(np.count_nonzero(self.node_active))

    @property
    def active_triangle_count(self) -> int:
        return int(np.count_nonzero(self.triangle_active))

    @property
    def active_quad_count(self) -> int:
        return int(np.count_nonzero(self.quad_active))

    @property
    def active_element_count(self) -> int:
        return self.active_triangle_count + self.active_quad_count

    @property
    def element_ids(self) -> np.ndarray:
        cached = self._cache.get("element_ids")
        if cached is None:
            cached = _freeze(np.concatenate((self.triangle_ids, self.quad_ids)), np.int64)
            self._cache["element_ids"] = cached
        return cached

    @property
    def element_active(self) -> np.ndarray:
        cached = self._cache.get("element_active")
        if cached is None:
            cached = _freeze(np.concatenate((self.triangle_active, self.quad_active)), bool)
            self._cache["element_active"] = cached
        return cached

    @property
    def element_owner_handles(self) -> np.ndarray:
        cached = self._cache.get("element_owner_handles")
        if cached is None:
            cached = _freeze(
                np.concatenate((self.triangle_owner_handles, self.quad_owner_handles)), np.int32
            )
            self._cache["element_owner_handles"] = cached
        return cached

    @property
    def order(self) -> str:
        widths: list[int] = []
        if self.num_triangles:
            widths.append(self.triangle_connectivity.shape[1])
        if self.num_quads:
            widths.append(self.quad_connectivity.shape[1])
        return "quadratic" if any(width in (6, 8) for width in widths) else "linear"

    @property
    def is_quadratic(self) -> bool:
        return self.order == "quadratic"

    def _row_for_id(self, kind: str, stable_id: int) -> int:
        key = f"{kind}_id_rows"
        lookup = self._cache.get(key)
        if lookup is None:
            ids = {
                "node": self.node_ids,
                "triangle": self.triangle_ids,
                "quad": self.quad_ids,
            }[kind]
            lookup = {int(value): row for row, value in enumerate(ids)}
            self._cache[key] = lookup
        try:
            return int(lookup[stable_id])
        except KeyError:
            raise KeyError(stable_id) from None

    @property
    def nodes(self) -> Mapping[int, np.ndarray]:
        return self._cache.setdefault("nodes_view", _CoreMapping(self, "node"))

    @property
    def tris(self) -> Mapping[int, tuple[int, ...]]:
        return self._cache.setdefault("tris_view", _CoreMapping(self, "triangle"))

    @property
    def quads(self) -> Mapping[int, tuple[int, ...]]:
        return self._cache.setdefault("quads_view", _CoreMapping(self, "quad"))

    @property
    def shells(self) -> Mapping[int, tuple[int, ...]]:
        return self._cache.setdefault("shells_view", _ShellMapping(self))

    @property
    def compatibility(self) -> MeshCompatibilityView:
        return self._cache.setdefault("compatibility_view", MeshCompatibilityView(self))

    @property
    def legacy(self) -> MeshCompatibilityView:
        return self.compatibility

    def corners_of(self, element_id: int) -> tuple[int, ...]:
        try:
            row = self._row_for_id("triangle", int(element_id))
            if self.triangle_active[row]:
                return tuple(int(self.node_ids[index]) for index in self.triangle_connectivity[row, :3])
        except KeyError:
            pass
        try:
            row = self._row_for_id("quad", int(element_id))
            if self.quad_active[row]:
                return tuple(int(self.node_ids[index]) for index in self.quad_connectivity[row, :4])
        except KeyError:
            pass
        raise MeshError(f"no active shell element {element_id}")

    def node_positions(self) -> np.ndarray:
        rows = np.flatnonzero(self.node_active)
        rows = rows[np.argsort(self.node_ids[rows], kind="stable")]
        return np.array(self.node_coordinates[rows], copy=True)

    def owner(self, handle: int) -> Any | None:
        return None if int(handle) < 0 else self.owner_table[int(handle)]

    def owner_of_node(self, node_id: int) -> Any | None:
        return self.owner(int(self.node_owner_handles[self._row_for_id("node", int(node_id))]))

    def owner_of_element(self, element_id: int) -> Any | None:
        try:
            return self.owner(int(self.triangle_owner_handles[self._row_for_id("triangle", int(element_id))]))
        except KeyError:
            return self.owner(int(self.quad_owner_handles[self._row_for_id("quad", int(element_id))]))

    def _active_elements(self) -> Iterator[tuple[int, np.ndarray]]:
        for row in np.flatnonzero(self.triangle_active):
            yield int(row), self.triangle_connectivity[row]
        offset = self.num_triangles
        for row in np.flatnonzero(self.quad_active):
            yield offset + int(row), self.quad_connectivity[row]

    @property
    def node_to_element(self) -> CSRAdjacency:
        cached = self._cache.get("node_to_element")
        if cached is None:
            rows = [set() for _ in range(self.num_nodes)]
            for element_row, connectivity in self._active_elements():
                for node in connectivity:
                    rows[int(node)].add(element_row)
            cached = _csr(rows)
            self._cache["node_to_element"] = cached
        return cached

    @property
    def node_element_adjacency(self) -> CSRAdjacency:
        return self.node_to_element

    @property
    def node_to_node(self) -> CSRAdjacency:
        cached = self._cache.get("node_to_node")
        if cached is None:
            rows = [set() for _ in range(self.num_nodes)]
            for _, connectivity in self._active_elements():
                sequence = _boundary_sequence(connectivity)
                for first, second in zip(sequence, sequence[1:] + sequence[:1]):
                    rows[first].add(second)
                    rows[second].add(first)
            cached = _csr(rows)
            self._cache["node_to_node"] = cached
        return cached

    @property
    def node_adjacency(self) -> CSRAdjacency:
        return self.node_to_node

    @property
    def element_to_element(self) -> CSRAdjacency:
        cached = self._cache.get("element_to_element")
        if cached is None:
            rows = [set() for _ in range(self.num_elements)]
            incidence: dict[tuple[int, int], list[int]] = {}
            for element_row, connectivity in self._active_elements():
                for edge in corner_edges(connectivity):
                    incidence.setdefault(edge, []).append(element_row)
            for attached in incidence.values():
                for first in attached:
                    rows[first].update(second for second in attached if second != first)
            cached = _csr(rows)
            self._cache["element_to_element"] = cached
        return cached

    @property
    def element_adjacency(self) -> CSRAdjacency:
        return self.element_to_element

    @property
    def boundary_edges(self) -> np.ndarray:
        cached = self._cache.get("boundary_edges")
        if cached is None:
            incidence: dict[tuple[int, int], int] = {}
            for _, connectivity in self._active_elements():
                for edge in corner_edges(connectivity):
                    incidence[edge] = incidence.get(edge, 0) + 1
            values = sorted(edge for edge, count in incidence.items() if count == 1)
            cached = _freeze(np.asarray(values, dtype=np.int64).reshape((-1, 2)), np.int64)
            self._cache["boundary_edges"] = cached
        return cached

    def _element_lengths(self, connectivity: np.ndarray, active: np.ndarray) -> np.ndarray:
        result = np.zeros(connectivity.shape[0], dtype=np.float64)
        corner_count = 3 if connectivity.shape[1] in (3, 6) else 4
        for row in np.flatnonzero(active):
            corners = connectivity[row, :corner_count]
            lengths = [
                np.linalg.norm(
                    self.node_coordinates[corners[(index + 1) % corner_count]]
                    - self.node_coordinates[corners[index]]
                )
                for index in range(corner_count)
            ]
            result[row] = float(np.mean(lengths))
        return _freeze(result, np.float64)

    @property
    def triangle_characteristic_lengths(self) -> np.ndarray:
        cached = self._cache.get("triangle_characteristic_lengths")
        if cached is None:
            cached = self._element_lengths(self.triangle_connectivity, self.triangle_active)
            self._cache["triangle_characteristic_lengths"] = cached
        return cached

    @property
    def quad_characteristic_lengths(self) -> np.ndarray:
        cached = self._cache.get("quad_characteristic_lengths")
        if cached is None:
            cached = self._element_lengths(self.quad_connectivity, self.quad_active)
            self._cache["quad_characteristic_lengths"] = cached
        return cached

    @property
    def element_characteristic_lengths(self) -> np.ndarray:
        cached = self._cache.get("element_characteristic_lengths")
        if cached is None:
            cached = _freeze(
                np.concatenate((self.triangle_characteristic_lengths, self.quad_characteristic_lengths)),
                np.float64,
            )
            self._cache["element_characteristic_lengths"] = cached
        return cached

    @property
    def node_characteristic_lengths(self) -> np.ndarray:
        cached = self._cache.get("node_characteristic_lengths")
        if cached is None:
            result = np.zeros(self.num_nodes, dtype=np.float64)
            relation = self.node_to_element
            element_lengths = self.element_characteristic_lengths
            for row in np.flatnonzero(self.node_active):
                attached = relation[row]
                if attached.size:
                    result[row] = float(np.mean(element_lengths[attached]))
            cached = _freeze(result, np.float64)
            self._cache["node_characteristic_lengths"] = cached
        return cached

    @property
    def characteristic_lengths(self) -> np.ndarray:
        return self.node_characteristic_lengths

    @property
    def memory_bytes(self) -> int:
        arrays = (
            self.node_coordinates,
            self.triangle_connectivity,
            self.quad_connectivity,
            self.node_ids,
            self.triangle_ids,
            self.quad_ids,
            self.node_owner_handles,
            self.triangle_owner_handles,
            self.quad_owner_handles,
            self.node_active,
            self.triangle_active,
            self.quad_active,
        )
        return int(sum(array.nbytes for array in arrays))

    def with_activity(
        self,
        *,
        node_active: Any | None = None,
        triangle_active: Any | None = None,
        quad_active: Any | None = None,
    ) -> "MeshCore":
        return MeshCore(
            self.node_coordinates,
            self.triangle_connectivity,
            self.quad_connectivity,
            node_ids=self.node_ids,
            triangle_ids=self.triangle_ids,
            quad_ids=self.quad_ids,
            owner_table=self.owner_table,
            node_owner_handles=self.node_owner_handles,
            triangle_owner_handles=self.triangle_owner_handles,
            quad_owner_handles=self.quad_owner_handles,
            node_active=self.node_active if node_active is None else node_active,
            triangle_active=self.triangle_active if triangle_active is None else triangle_active,
            quad_active=self.quad_active if quad_active is None else quad_active,
        )

    def deactivate_elements(self, element_ids: Sequence[int]) -> "MeshCore":
        triangles = np.array(self.triangle_active, copy=True)
        quads = np.array(self.quad_active, copy=True)
        for stable_id in element_ids:
            try:
                triangles[self._row_for_id("triangle", int(stable_id))] = False
                continue
            except KeyError:
                pass
            try:
                quads[self._row_for_id("quad", int(stable_id))] = False
            except KeyError:
                raise MeshError(f"no element {stable_id}") from None
        return self.with_activity(triangle_active=triangles, quad_active=quads)

    def compact(self, *, drop_unused_nodes: bool = True) -> "MeshCore":
        triangle_rows = np.flatnonzero(self.triangle_active)
        quad_rows = np.flatnonzero(self.quad_active)
        keep_nodes = np.array(self.node_active, copy=True)
        if drop_unused_nodes:
            keep_nodes[:] = False
            if triangle_rows.size:
                keep_nodes[self.triangle_connectivity[triangle_rows].ravel()] = True
            if quad_rows.size:
                keep_nodes[self.quad_connectivity[quad_rows].ravel()] = True
        node_rows = np.flatnonzero(keep_nodes)
        remap = np.full(self.num_nodes, -1, dtype=np.int64)
        remap[node_rows] = np.arange(node_rows.size, dtype=np.int64)
        triangles = remap[self.triangle_connectivity[triangle_rows]]
        quads = remap[self.quad_connectivity[quad_rows]]
        return MeshCore(
            self.node_coordinates[node_rows],
            triangles,
            quads,
            node_ids=self.node_ids[node_rows],
            triangle_ids=self.triangle_ids[triangle_rows],
            quad_ids=self.quad_ids[quad_rows],
            owner_table=self.owner_table,
            node_owner_handles=self.node_owner_handles[node_rows],
            triangle_owner_handles=self.triangle_owner_handles[triangle_rows],
            quad_owner_handles=self.quad_owner_handles[quad_rows],
        )

    @classmethod
    def from_id_connectivity(
        cls,
        node_coordinates: Any,
        *,
        node_ids: Any,
        triangles: Any | None = None,
        quadrilaterals: Any | None = None,
        **kwargs: Any,
    ) -> "MeshCore":
        ids = np.asarray(node_ids, dtype=np.int64)
        lookup = {int(value): row for row, value in enumerate(ids)}

        def convert(value: Any | None) -> Any | None:
            if value is None:
                return None
            array = np.asarray(value, dtype=np.int64)
            try:
                return np.vectorize(lambda item: lookup[int(item)], otypes=[np.int64])(array)
            except KeyError as exc:
                raise MeshError(f"connectivity references unknown node ID {exc.args[0]}") from None

        return cls(
            node_coordinates,
            convert(triangles),
            convert(quadrilaterals),
            node_ids=ids,
            **kwargs,
        )

