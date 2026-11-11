"""True local remesh of a changed region with the retained outer cells exact.

Contract
--------
``remesh_region`` takes an old :class:`~anymesher.mesh.Mesh`, the face whose
element provenance is ``mesh.elements_of_face[face_id]``, and the changed region
(explicit element ids and/or footprint polygons).  Every cell touched by the
change is replaced as a *complete cell*; the boundary of the replaced union is
the set of exact fixed stations.  This module only *produces* a staged
replacement (kernel patch, geometry/topology/coverage proof, association
staging).  The candidate mesh itself is published by the owner function
``regional_update.assemble_regional_replacement`` (frozen outside, interface
conformity, association rebuild, metadata refusal, explicit quality), so there is
no second assembly implementation here.  Nothing is published unless every gate
passes; the caller's mesh is never mutated.

Scope actually admitted
-----------------------
* Planar faces only (``surface_kind == "plane"`` *and* every face node within the
  caller-supplied owner planarity tolerance of the face plane), meshed by the
  owner planar kernel (``surface_mesh.mesh_planar_surface`` with a seeded
  boundary) or an explicitly injected kernel (reported as
  ``kernel_injected``).  Linear (Q4/T3) and quadratic (Q8/T6).  Straight edges.
* Every curved family fails closed (``CURVED_FAMILY_UNIMPLEMENTED``): no owner
  chart/source-correspondence registry exists to prove one, and a caller string is
  never authority.  ``held_case`` is only an additional refusal, never a guard.

Mandatory for ``status == "accepted"``: geometry model/revision on the mesh,
planarity tolerance, ``RegionalQualityLimits``, a ``source_check`` (source
geometry correspondence proof), valid order, full coverage proof.

Coverage compares the exact signed areas of the represented planar polygons.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import MeshError
from .mesh import Mesh
from .regional_update import (
    RegionalQualityLimits,
    _element_nodes,
    _validate_structure,
    assemble_regional_replacement,
)

__all__ = [
    "HELD_CASES",
    "KernelInput",
    "KernelOutput",
    "NewDomain",
    "PatchLoop",
    "PatchStation",
    "PatchTolerances",
    "RegionBoundary",
    "RegionalPatchRefusal",
    "RegionalPatchReport",
    "RegionalPatchRequest",
    "RegionalPatchResult",
    "owner_planar_kernel",
    "remesh_region",
    "try_remesh_region",
    "verify_regional_patch",
]

_INT64_MAX = 2**63 - 1

HELD_CASES = (
    "two_split_quarter_cylinders",
    "principal_dome_cut_nurbs",
    "four_face_spline",
)

_CURVED_FAMILIES = frozenset({
    "cylinder", "cone", "sphere", "torus", "nurbs", "spline", "bspline",
    "ruled", "extrusion", "revolution", "dome", "coons",
})

# Only a diagnosed fixed-boundary defect may enlarge the region.  Kernel,
# quality, owner-conformity and source failures are never retried by growth.
_GROWABLE = frozenset({"ODD_BOUNDARY_PARITY", "REGION_PINCH", "REGION_DISCONNECTED_OUTER"})


class RegionalPatchRefusal(MeshError):
    """Typed refusal; the caller falls back (e.g. to a full-face remesh)."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = dict(details)


# ----------------------------------------------------------------- data types


@dataclass(frozen=True)
class PatchTolerances:
    """Owner-supplied tolerances.  ``planarity`` is required for acceptance."""

    planarity: float | None = None


@dataclass(frozen=True)
class PatchStation:
    xyz: tuple[float, float, float]
    node_id: int | None = None  # existing fixed node, or None for a new node
    edge_id: int | None = None  # informational geometry association
    uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class PatchLoop:
    stations: tuple[PatchStation, ...]
    # Optional XYZ of a curved segment midside (segment i joins station i, i+1).
    segment_midsides: tuple[tuple[float, float, float] | None, ...] | None = None
    straight: bool = True
    interface: tuple[bool, ...] | None = None  # derived loops only


@dataclass(frozen=True)
class NewDomain:
    outer: PatchLoop
    holes: tuple[PatchLoop, ...] = ()


@dataclass(frozen=True)
class RegionBoundary:
    """Derived boundary of one connected replaced region (outer CCW, holes CW)."""

    component: int
    element_ids: tuple[int, ...]
    outer: PatchLoop
    holes: tuple[PatchLoop, ...]


@dataclass(frozen=True)
class KernelInput:
    outer: np.ndarray  # (n, 2) fixed stations, CCW
    holes: tuple[np.ndarray, ...]  # CW fixed stations
    target_size: float
    recombine: bool = True
    order: str = "linear"
    sizing_zones: tuple[tuple[np.ndarray, float], ...] = ()  # chart polygon and requested edge size


@dataclass(frozen=True)
class KernelOutput:
    points: np.ndarray  # (m, 2); fixed stations must be reproduced exactly
    quads: tuple[tuple[int, ...], ...]  # rows into points
    tris: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class RegionalPatchRequest:
    old_mesh: Mesh
    face_id: int
    surface_kind: str = "plane"
    replace_element_ids: Sequence[int] | None = None
    footprints: Sequence[Sequence[Sequence[float]]] = ()  # changed footprints (xyz polygons)
    target_size: float | None = None  # None: mean retained-boundary edge length
    new_domain: NewDomain | None = None  # explicit; single component only
    domain_builder: Callable[[RegionBoundary], NewDomain] | None = None
    added_holes: Sequence[PatchLoop] = ()
    added_hole_edge_ids: Sequence[int] = ()  # geometry edge id per added hole (required)
    drop_hole_node_ids: Sequence[int] = ()  # any node id on a free hole loop
    retired_geometry_keys: Mapping[str, Sequence[int]] | None = None  # owner retire of removed holes
    remove_mapped_grid: bool = False  # consent to drop grid_of_face/block grids of this face
    discard_stale_diagnostics: bool = False
    require_all_quad: bool = True
    max_growth_rings: int = 2
    kernel: Callable[[KernelInput], KernelOutput] | None = None  # injected (test/special) kernel
    held_case: str | None = None
    tolerances: PatchTolerances = PatchTolerances()
    quality: RegionalQualityLimits | None = None  # REQUIRED (owner explicit limits)
    quality_gate: Callable[[Mesh, Sequence[int]], bool] | None = None  # extra, optional
    source_check: Callable[[Mesh, Mesh, "RegionalPatchReport"], bool] | None = None  # REQUIRED
    node_high_water: int | None = None  # highest node id ever issued (survives delete/undo)
    element_high_water: int | None = None
    node_id_floor: int | None = None  # lowest id a new node may take
    element_id_floor: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    # Each edited size field supplies its physical footprint and local edge target.
    # Acceptance requires a fresh quadrilateral meeting that size in each zone.
    sizing_zones: Sequence[tuple[Sequence[Sequence[float]], float]] = ()


@dataclass(frozen=True)
class RegionalPatchReport:
    status: str
    face_id: int
    route: str
    order: str
    kernel_injected: bool
    replaced_element_ids: tuple[int, ...]
    new_element_ids: tuple[int, ...]
    new_node_ids: tuple[int, ...]
    removed_node_ids: tuple[int, ...]
    retained_node_ids: tuple[int, ...]
    retained_element_ids: tuple[int, ...]
    interface_node_ids: tuple[int, ...]
    renumbered_free_nodes: tuple[tuple[int, int], ...]  # (old id, new id), free-boundary only
    dropped_geometry_nodes: Mapping[str, tuple[int, ...]]
    removed_grids: tuple[int, ...]
    new_hole_edges: Mapping[int, tuple[int, ...]]
    discarded_diagnostic_keys: tuple[str, ...]
    node_high_water: int  # monotonic high water after this operation
    element_high_water: int
    quality: Mapping[str, Any] | None
    retained_element_count: int
    components: int
    boundary_station_count: int
    all_quad: bool
    growth_rings: int
    attempts: tuple[Mapping[str, Any], ...]
    checks: tuple[str, ...]
    unchecked: tuple[str, ...]
    target_size: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RegionalPatchResult:
    mesh: Mesh
    report: RegionalPatchReport
    new_node_uv: Mapping[int, tuple[float, float]]


# ------------------------------------------------------------------- geometry


def _ekey(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _corner_ids(conn: Sequence[int]) -> tuple[int, ...]:
    count = len(conn)
    if count in (4, 8):
        return tuple(int(v) for v in conn[:4])
    if count in (3, 6):
        return tuple(int(v) for v in conn[:3])
    raise RegionalPatchRefusal("UNSUPPORTED_ELEMENT", f"shell width {count}")


def _area2(xy) -> float:
    """Signed polygon area (positive for counter-clockwise)."""

    arr = np.asarray(xy, dtype=float)
    x, y = arr[:, 0], arr[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient(a, b, c) -> int:
    from .triangulation import orient2d
    v = orient2d(a, b, c)
    return int(v > 0.0) - int(v < 0.0)


def _exact_double_area(polygon):
    ring = [(Fraction(float(x)), Fraction(float(y))) for x, y in polygon]
    return sum(x * yy - y * xx for (x, y), (xx, yy) in zip(ring, ring[1:] + ring[:1]))


def _on_seg(p, a, b) -> bool:
    return (_orient(a, b, p) == 0 and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _strict_inside(p, poly) -> bool:
    n = len(poly)
    for i in range(n):
        if _on_seg(p, poly[i], poly[(i + 1) % n]):
            return False
    inside = False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x > p[0]:
                inside = not inside
    return inside


def _centroid(poly):
    return (sum(float(p[0]) for p in poly) / len(poly), sum(float(p[1]) for p in poly) / len(poly))


def _overlaps(cell, poly) -> bool:
    """Interior overlap by strict sign tests; touching only is not overlap."""

    n, m = len(cell), len(poly)
    for i in range(n):
        p1, p2 = cell[i], cell[(i + 1) % n]
        for j in range(m):
            q1, q2 = poly[j], poly[(j + 1) % m]
            if (_orient(p1, p2, q1) * _orient(p1, p2, q2) < 0
                    and _orient(q1, q2, p1) * _orient(q1, q2, p2) < 0):
                return True
    return (any(_strict_inside(v, poly) for v in cell)
            or any(_strict_inside(v, cell) for v in poly)
            or _strict_inside(_centroid(cell), poly)
            or _strict_inside(_centroid(poly), cell))


def _cell_positive(xy: Sequence[Sequence[float]]) -> bool:
    n = len(xy)
    return all(_orient(xy[i], xy[(i + 1) % n], xy[(i + 2) % n]) > 0 for i in range(n))


class _Chart:
    """Plane working frame of the face: xyz <-> (x, y)."""

    def __init__(self, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray, normal: np.ndarray):
        self.origin, self.e1, self.e2, self.normal = origin, e1, e2, normal

    def to2(self, xyz) -> np.ndarray:
        d = np.asarray(xyz, dtype=float) - self.origin
        return np.array([float(d @ self.e1), float(d @ self.e2)])

    def to3(self, xy) -> np.ndarray:
        return self.origin + float(xy[0]) * self.e1 + float(xy[1]) * self.e2

    def height(self, xyz) -> float:
        return float((np.asarray(xyz, dtype=float) - self.origin) @ self.normal)


def _plane_chart(mesh: Mesh, element_id: int) -> _Chart:
    pts = np.array([mesh.nodes[n] for n in mesh.corners_of(element_id)], dtype=float)
    normal = np.zeros(3)
    for i in range(len(pts)):
        normal += np.cross(pts[i], pts[(i + 1) % len(pts)])
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        raise RegionalPatchRefusal("DEGENERATE_CELL", f"element {element_id} has no area")
    normal = normal / length
    edge = pts[1] - pts[0]
    e1 = edge - normal * float(edge @ normal)
    elen = float(np.linalg.norm(e1))
    if elen == 0.0:
        raise RegionalPatchRefusal("DEGENERATE_CELL", f"element {element_id} has a zero edge")
    e1 = e1 / elen
    return _Chart(pts[0].copy(), e1, np.cross(normal, e1), normal)


class _Ids:
    """Globally monotonic allocator: never reuses an id at or below a high water."""

    def __init__(self, mesh: Mesh, request: RegionalPatchRequest) -> None:
        def given(value, name):
            if value is None:
                return 0
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
                    or not 0 <= int(value) <= _INT64_MAX:
                raise RegionalPatchRefusal("BAD_ID_FLOOR", f"{name} must be an int64 >= 0, not {value!r}")
            return int(value)

        used = [0]
        for table in (mesh.quads, mesh.tris, mesh.beams, mesh.couplings):
            used.extend(int(k) for k in table)
        n_hw = max(given(request.node_high_water, "node_high_water"), max(mesh.nodes, default=0))
        e_hw = max(given(request.element_high_water, "element_high_water"), max(used))
        self.node_hw, self.elem_hw = n_hw, e_hw
        self.node = max(n_hw + 1, given(request.node_id_floor, "node_id_floor"), 1)
        self.elem = max(e_hw + 1, given(request.element_id_floor, "element_id_floor"), 1)

    def next_node(self) -> int:
        if self.node > _INT64_MAX:
            raise RegionalPatchRefusal("ID_EXHAUSTED", "node ids exhausted (int64)")
        self.node_hw = self.node
        self.node += 1
        return self.node_hw

    def next_elem(self) -> int:
        if self.elem > _INT64_MAX:
            raise RegionalPatchRefusal("ID_EXHAUSTED", "element ids exhausted (int64)")
        self.elem_hw = self.elem
        self.elem += 1
        return self.elem_hw


# ------------------------------------------------------------------ kernels


def owner_planar_kernel(kin: KernelInput) -> KernelOutput:
    """The owner native planar mesher with the boundary stations kept as given."""

    if kin.recombine and not kin.holes:
        mapped = _mapped_station_grid(kin)
        if mapped is not None:
            return mapped
    if kin.recombine:
        # Certified full station grid: the fixed stations themselves define the
        # cells (target_size is not used to refine; every rim station is kept).
        aligned = _aligned_station_grid(kin)
        if aligned is not None:
            return aligned
    from .surface_mesh import mesh_planar_surface

    core = mesh_planar_surface(
        kin.outer, tuple(kin.holes), (),
        recombine=kin.recombine, order=kin.order, target_size=kin.target_size,
        _boundary_is_seeded=True,
    )
    ids = sorted(int(i) for i in core.nodes)
    row = {i: k for k, i in enumerate(ids)}
    pts = np.array([np.asarray(core.nodes[i], dtype=float)[:2] for i in ids])
    quads = tuple(tuple(row[int(n)] for n in core.quads[q][:4]) for q in sorted(core.quads))
    tris = tuple(tuple(row[int(n)] for n in core.tris[t][:3]) for t in sorted(core.tris))
    return KernelOutput(pts, quads, tris)


def _mapped_station_grid(kin: KernelInput) -> KernelOutput | None:
    """Use the existing mapped generator for compatible straight-sided patches."""
    from .mapped import coons_grid

    ring = np.asarray(kin.outer)
    turns = [_orient(ring[i - 1], ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    corners = [i for i, turn in enumerate(turns) if turn]
    if len(corners) != 4 or any(turn < 0 for turn in turns):
        return None
    closed = np.concatenate((ring, ring))
    sides = [closed[a:b + 1] for a, b in zip(corners, corners[1:] + [corners[0] + len(ring)])]
    a, b, c, d = sides
    if len(a) != len(c) or len(b) != len(d):
        return None
    grid = coons_grid(*(np.column_stack((side, np.zeros(len(side)))) for side in (a, b, c[::-1], d[::-1])))
    # The original station bytes are authoritative even if blending introduces rounding.
    grid[:, 0, :2], grid[-1, :, :2] = a, b
    grid[:, -1, :2], grid[0, :, :2] = c[::-1], d[::-1]
    width = len(b)
    cells = tuple((i * width + j, (i + 1) * width + j, (i + 1) * width + j + 1, i * width + j + 1)
                  for i in range(len(a) - 1) for j in range(width - 1))
    points = grid[:, :, :2].reshape((-1, 2))
    longest = max(float(np.linalg.norm(np.diff(grid, axis=i), axis=2).max()) for i in (0, 1))
    local = [(polygon, size) for polygon, size in kin.sizing_zones if _overlaps(ring, polygon)]
    target = min([kin.target_size, *(size for _, size in local)])
    if longest <= target:
        return KernelOutput(points, cells)

    # A homothetic copy of the complete grid forms the finer core. Join each
    # outer directed segment to its inner image with one quad; no additional
    # corner or midside station is inserted on the parent boundary.
    center = np.mean(ring[corners], axis=0)
    if local:
        requested_center = np.mean([_centroid(polygon) for polygon, _ in local], axis=0)
        if _strict_inside(requested_center, ring):
            center = requested_center
    # The largest dyadic scale meeting the target gives a nonzero transition
    # width even when the requested size is just below the original spacing.
    ratio = target / longest
    if not 0 < ratio < 1:
        raise RegionalPatchRefusal("LOCAL_SIZE_UNMET", "target is outside representable mapped grid scales")
    scale = math.ldexp(1.0, math.frexp(ratio)[1] - 1)
    inner = center + scale * (points - center)
    ids = np.arange(len(points)).reshape((len(a), width))
    boundary = np.concatenate((ids[:, 0], ids[-1, 1:], ids[-2::-1, -1], ids[0, -2:0:-1]))
    outer_start = len(points)
    transitions = tuple(
        (outer_start + i, outer_start + (i + 1) % len(boundary), int(boundary[(i + 1) % len(boundary)]), int(j))
        for i, j in enumerate(boundary)
    )
    return KernelOutput(np.vstack((inner, points[boundary])), cells + transitions)


def _axis_rect(ring, ccw: bool):
    """(x0, x1, y0, y1) if ``ring`` is an exactly axis-aligned rectangle loop, else None."""

    pts = [(float(p[0]), float(p[1])) for p in np.asarray(ring, dtype=float)]
    if len(pts) < 4 or len(set(pts)) != len(pts):
        return None
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    if not (x0 < x1 and y0 < y1):
        return None
    if any(p[0] not in (x0, x1) and p[1] not in (y0, y1) for p in pts):
        return None
    if not {(x0, y0), (x1, y0), (x1, y1), (x0, y1)} <= set(pts):
        return None
    area = _exact_double_area(pts)
    if abs(area) != 2 * (Fraction(x1) - Fraction(x0)) * (Fraction(y1) - Fraction(y0)):
        return None
    if (area > 0) != ccw:
        return None
    return x0, x1, y0, y1


def _aligned_station_grid(kin: KernelInput) -> KernelOutput | None:
    """All-quad patch from the full station grid of an axis-aligned rectangle.

    Admitted only when the outer loop is an axis-aligned rectangle whose side
    stations define the grid lines, and every hole is an axis-aligned rectangle
    strictly inside, with corners on grid lines and exactly the grid points of its
    perimeter as stations, holes strictly separated.  Whole grid cells inside a hole
    are clipped; the resulting directed boundary must equal the outer ring and the
    reversed hole rings exactly.  Anything else returns ``None`` (not claimed).
    """

    outer = _axis_rect(kin.outer, True)
    if outer is None:
        return None
    x0, x1, y0, y1 = outer
    ring = [(float(p[0]), float(p[1])) for p in np.asarray(kin.outer, dtype=float)]
    xs = sorted({p[0] for p in ring if p[1] == y0})
    ys = sorted({p[1] for p in ring if p[0] == x0})
    xi = {v: k for k, v in enumerate(xs)}
    yi = {v: k for k, v in enumerate(ys)}
    boxes = []
    hole_rings = []
    for hole in kin.holes:
        box = _axis_rect(hole, False)
        if box is None:
            return None
        hx0, hx1, hy0, hy1 = box
        if not (x0 < hx0 and hx1 < x1 and y0 < hy0 and hy1 < y1):
            return None
        if hx0 not in xi or hx1 not in xi or hy0 not in yi or hy1 not in yi:
            return None
        want = {(x, y) for x in xs for y in ys
                if hx0 <= x <= hx1 and hy0 <= y <= hy1 and (x in (hx0, hx1) or y in (hy0, hy1))}
        pts = [(float(p[0]), float(p[1])) for p in np.asarray(hole, dtype=float)]
        if set(pts) != want or len(pts) != len(want):
            return None
        for ob in boxes:
            if not (hx1 < ob[0] or ob[1] < hx0 or hy1 < ob[2] or ob[3] < hy0):
                return None
        boxes.append((hx0, hx1, hy0, hy1))
        hole_rings.append(pts)
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if any(b[0] <= xs[i] and xs[i + 1] <= b[1] and b[2] <= ys[j] and ys[j + 1] <= b[3]
                   for b in boxes):
                continue
            cells.append(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
    if not cells:
        return None
    directed = set()
    for cell in cells:
        for k in range(4):
            directed.add((cell[k], cell[(k + 1) % 4]))
    bnd = {e for e in directed if (e[1], e[0]) not in directed}
    expected = set()
    for lp in (ring, *hole_rings):
        try:
            idx = [(xi[p[0]], yi[p[1]]) for p in lp]
        except KeyError:
            return None
        expected |= {(idx[k], idx[(k + 1) % len(idx)]) for k in range(len(idx))}
    if bnd != expected:
        return None
    used = sorted({v for cell in cells for v in cell})
    row = {v: k for k, v in enumerate(used)}
    points = np.array([(xs[i], ys[j]) for i, j in used], dtype=float)
    return KernelOutput(points, tuple(tuple(row[v] for v in cell) for cell in cells))


# --------------------------------------------------------------- region logic


def _shells(mesh: Mesh) -> dict[int, tuple[int, ...]]:
    return {int(k): tuple(int(n) for n in v) for k, v in {**mesh.quads, **mesh.tris}.items()}


def _edge_users(shells) -> dict[tuple[int, int], list[int]]:
    users: dict[tuple[int, int], list[int]] = {}
    for eid, conn in shells.items():
        cs = _corner_ids(conn)
        for k in range(len(cs)):
            users.setdefault(_ekey(cs[k], cs[(k + 1) % len(cs)]), []).append(eid)
    return users


def _old_mids(shells) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for conn in shells.values():
        if len(conn) in (6, 8):
            cs = _corner_ids(conn)
            for k in range(len(cs)):
                out[_ekey(cs[k], cs[(k + 1) % len(cs)])] = conn[len(cs) + k]
    return out


def _select_region(request, chart, shells, face_set) -> list[int]:
    ids: list[int] = []
    if request.replace_element_ids is not None:
        ids = sorted({int(i) for i in request.replace_element_ids})
        for i in ids:
            if i not in face_set:
                raise RegionalPatchRefusal("REPLACE_NOT_IN_FACE", f"element {i} not in face provenance")
    polys = [[tuple(chart.to2(p)) for p in poly] for poly in request.footprints]
    if polys:
        extra = set(ids)
        for eid in sorted(face_set):
            cell = [tuple(chart.to2(request.old_mesh.nodes[n])) for n in _corner_ids(shells[eid])]
            if any(_overlaps(cell, poly) for poly in polys):
                extra.add(eid)
        ids = sorted(extra)
    if not ids:
        raise RegionalPatchRefusal("EMPTY_REGION", "no changed region selected")
    return ids


def _components(shells, region: Sequence[int]) -> list[list[int]]:
    owner: dict[tuple[int, int], list[int]] = {}
    for eid in region:
        cs = _corner_ids(shells[eid])
        for k in range(len(cs)):
            owner.setdefault(_ekey(cs[k], cs[(k + 1) % len(cs)]), []).append(eid)
    seen: set[int] = set()
    out = []
    for start in sorted(region):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            cs = _corner_ids(shells[cur])
            for k in range(len(cs)):
                for other in owner[_ekey(cs[k], cs[(k + 1) % len(cs)])]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        out.append(sorted(comp))
    return out


def _region_boundary(mesh, shells, users, chart, comp, index) -> RegionBoundary:
    comp_set = set(comp)
    count: dict[tuple[int, int], int] = {}
    directed = []
    for eid in comp:
        cs = _corner_ids(shells[eid])
        for k in range(len(cs)):
            a, b = cs[k], cs[(k + 1) % len(cs)]
            count[_ekey(a, b)] = count.get(_ekey(a, b), 0) + 1
            directed.append((a, b))
    if any(c > 2 for c in count.values()):
        raise RegionalPatchRefusal("NONMANIFOLD_REGION", "an edge has more than two region cells")
    nxt: dict[int, int] = {}
    for a, b in directed:
        if count[_ekey(a, b)] == 1:
            if a in nxt:
                raise RegionalPatchRefusal("REGION_PINCH", f"region boundary pinches at node {a}")
            nxt[a] = b
    loops, seen = [], set()
    for start in sorted(nxt):
        if start in seen:
            continue
        loop, cur = [start], nxt[start]
        seen.add(start)
        while cur != start:
            if cur in seen or cur not in nxt:
                raise RegionalPatchRefusal("REGION_PINCH", f"region boundary is not a simple loop at {cur}")
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        loops.append(loop)
    outers, holes = [], []
    for loop in loops:
        xy = np.array([chart.to2(mesh.nodes[n]) for n in loop])
        stations = tuple(PatchStation(tuple(float(v) for v in mesh.nodes[n]), n) for n in loop)
        flags = tuple(
            any(u not in comp_set for u in users[_ekey(loop[i], loop[(i + 1) % len(loop)])])
            for i in range(len(loop))
        )
        item = PatchLoop(stations, interface=flags)
        (outers if _area2(xy) > 0 else holes).append(item)
    if len(outers) != 1:
        raise RegionalPatchRefusal("REGION_DISCONNECTED_OUTER", f"{len(outers)} outer loops in one component")
    holes.sort(key=lambda h: min(s.node_id for s in h.stations))
    return RegionBoundary(index, tuple(comp), outers[0], tuple(holes))


def _reverse_loop(loop: PatchLoop) -> PatchLoop:
    n = len(loop.stations)
    mids = None
    if loop.segment_midsides is not None:
        mids = tuple(loop.segment_midsides[(n - 2 - j) % n] for j in range(n))
    return PatchLoop(tuple(reversed(loop.stations)), mids, loop.straight, None)


def _bbox_overlap_pairs(a_xy: list[np.ndarray], b_xy: list[np.ndarray]):
    """Yield (i, j) whose bounding boxes overlap (strictly, so touching is skipped)."""

    if not a_xy or not b_xy:
        return
    bmin = np.array([c.min(axis=0) for c in b_xy])
    bmax = np.array([c.max(axis=0) for c in b_xy])
    for i, c in enumerate(a_xy):
        lo, hi = c.min(axis=0), c.max(axis=0)
        mask = np.all((bmin < hi) & (bmax > lo), axis=1)
        for j in np.nonzero(mask)[0]:
            yield i, int(j)


# --------------------------------------------------------------- one component


@dataclass
class _Patch:
    comp: list[int]
    new_nodes: dict[int, tuple[float, float, float]]
    new_uv: dict[int, tuple[float, float]]
    cells: list[tuple[int, ...]]  # global node ids (+ midsides when quadratic)
    boundary_keys: set[tuple[int, int]]
    stations: int
    renumbered: list[tuple[int, int]]
    hole_edges: dict[int, tuple[list[int], set[tuple[int, int]]]]


def _build_patch(request, mesh, users, old_mid, chart, comp, boundary, domain, shared,
                 retained_xy, ids: _Ids, size: float, kernel, quadratic: bool,
                 hole_edge_of_loop: Mapping[int, int], sizing_zones) -> _Patch:
    comp_set = set(comp)
    bnodes = {s.node_id for lp in (boundary.outer, *boundary.holes) for s in lp.stations}
    ikeys = set()
    for lp in (boundary.outer, *boundary.holes):
        n = len(lp.stations)
        for i in range(n):
            if lp.interface[i]:
                ikeys.add((lp.stations[i].node_id, lp.stations[(i + 1) % n].node_id))
    ipairs = {_ekey(a, b) for a, b in ikeys}
    tol = request.tolerances.planarity

    loops = []
    hole_edge_loops: dict[int, int] = {}  # edge id -> loop index
    for idx, lp in enumerate((domain.outer, *domain.holes)):
        if len(lp.stations) < 3:
            raise RegionalPatchRefusal("DOMAIN_DEGENERATE", "a loop needs at least three stations")
        for s in lp.stations:
            if abs(chart.height(s.xyz)) > tol:
                raise RegionalPatchRefusal("STATION_OFF_PLANE", f"station {s.xyz} is off the face plane")
        edge_id = hole_edge_of_loop.get(id(lp))
        xy = np.array([chart.to2(s.xyz) for s in lp.stations])
        area = _area2(xy)
        if area == 0.0:
            raise RegionalPatchRefusal("DOMAIN_DEGENERATE", "a loop has zero area")
        if (area > 0) != (idx == 0):
            lp = _reverse_loop(lp)
            xy = xy[::-1].copy()
        if edge_id is not None:
            hole_edge_loops[edge_id] = idx
        loops.append((lp, xy))

    seen_ids, seen_xy = set(), set()
    for lp, xy in loops:
        for s, p in zip(lp.stations, xy):
            if s.node_id is not None:
                if s.node_id in seen_ids:
                    raise RegionalPatchRefusal("DOMAIN_DEGENERATE", f"node {s.node_id} used twice")
                seen_ids.add(s.node_id)
                if s.node_id not in bnodes:
                    raise RegionalPatchRefusal("INTERFACE_MISMATCH", f"prescribed node {s.node_id} does not lie on the region boundary")
                if not np.array_equal(np.asarray(s.xyz, dtype=float), np.asarray(mesh.nodes[s.node_id], dtype=float)):
                    raise RegionalPatchRefusal("STATION_MOVED", f"node {s.node_id} xyz differs from the old mesh")
            key = (float(p[0]), float(p[1]))
            if key in seen_xy:
                raise RegionalPatchRefusal("DOMAIN_DEGENERATE", "duplicate station coordinates")
            seen_xy.add(key)

    # Added holes must lie strictly inside the outer loop and outside other holes.
    outer_poly = [tuple(p) for p in loops[0][1]]
    for edge_id, idx in hole_edge_loops.items():
        for p in loops[idx][1]:
            if not _strict_inside(tuple(p), outer_poly) or any(
                    _strict_inside(tuple(p), [tuple(q) for q in oxy])
                    for j, (_, oxy) in enumerate(loops[1:], start=1) if j != idx):
                raise RegionalPatchRefusal("ADDED_HOLE_OUTSIDE_REGION", "hole station not strictly inside the domain")

    # Interface preservation: exact directed segments, no foreign retained edges.
    dseg = set()
    for lp, _ in loops:
        n = len(lp.stations)
        for i in range(n):
            a, b = lp.stations[i].node_id, lp.stations[(i + 1) % n].node_id
            dseg.add((a, b))
            if a is not None and b is not None:
                k = _ekey(a, b)
                if (a, b) not in ikeys and k in users and any(u not in comp_set for u in users[k]):
                    raise RegionalPatchRefusal("INTERFACE_MISMATCH", f"segment {a}-{b} is a retained edge")
    missing = ikeys - dseg
    if missing:
        raise RegionalPatchRefusal("INTERFACE_NOT_PRESERVED", f"retained interface segments dropped: {sorted(missing)[:4]}")

    total = sum(len(lp.stations) for lp, _ in loops)
    if request.require_all_quad and (total % 2 or total < 4):
        raise RegionalPatchRefusal("ODD_BOUNDARY_PARITY", f"{total} boundary segments cannot be all-quad")

    stations, rows_seg, bases = [], [], []
    for lp, xy in loops:
        base = len(stations)
        bases.append(base)
        stations.extend(zip(lp.stations, xy))
        n = len(lp.stations)
        rows_seg.extend((base + i, base + (i + 1) % n) for i in range(n))
    kin = KernelInput(loops[0][1].copy(), tuple(xy.copy() for _, xy in loops[1:]), size, True, "linear",
                      tuple((polygon.copy(), target) for polygon, target in sizing_zones))
    try:
        out = kernel(kin)
    except RegionalPatchRefusal:
        raise
    except Exception as exc:  # the owner kernel reports failure by exception
        raise RegionalPatchRefusal("KERNEL_FAILED", f"{type(exc).__name__}: {exc}") from exc

    pts = np.asarray(out.points, dtype=float)
    lookup: dict[tuple[float, float], int] = {}
    for r, p in enumerate(pts):
        k = (float(p[0]), float(p[1]))
        if k in lookup:
            raise RegionalPatchRefusal("KERNEL_FAILED", "kernel returned duplicate points")
        lookup[k] = r
    station_row = []
    for _, p in stations:
        r = lookup.get((float(p[0]), float(p[1])))
        if r is None:
            raise RegionalPatchRefusal("KERNEL_MOVED_FIXED_STATION", "kernel did not reproduce a fixed station exactly")
        station_row.append(r)
    if request.require_all_quad and out.tris:
        raise RegionalPatchRefusal("NON_QUAD_PATCH", f"{len(out.tris)} triangles in an all-quad patch")

    cells = []
    for cell in (*out.quads, *out.tris):
        cell = tuple(int(r) for r in cell)
        if len(cell) not in (3, 4) or not all(0 <= r < len(pts) for r in cell):
            raise RegionalPatchRefusal("KERNEL_FAILED", "malformed kernel cell")
        if _area2(pts[list(cell)]) < 0:
            cell = tuple(reversed(cell))
        if not _cell_positive([tuple(pts[r]) for r in cell]):
            raise RegionalPatchRefusal("NONPOSITIVE_CELL", "kernel cell is inverted, degenerate or non-convex")
        cells.append(cell)
    if not cells:
        raise RegionalPatchRefusal("PATCH_TOPOLOGY", "kernel produced no cells")
    directed = set()
    for cell in cells:
        for k in range(len(cell)):
            e = (cell[k], cell[(k + 1) % len(cell)])
            if e in directed:
                raise RegionalPatchRefusal("PATCH_TOPOLOGY", "directed edge used twice")
            directed.add(e)
    bnd = {e for e in directed if (e[1], e[0]) not in directed}
    expected = {(station_row[a], station_row[b]) for a, b in rows_seg}
    if bnd != expected:
        raise RegionalPatchRefusal("PATCH_BOUNDARY_MISMATCH", "patch boundary differs from the fixed stations (coverage)")

    # Interior coverage proof: no overlap among cells, no cell node strictly in a
    # hole or outside the outer loop, areas sum to the domain area, and no overlap
    # with any retained cell of the face.
    cxy = [pts[list(c)] for c in cells]
    for i, j in _bbox_overlap_pairs(cxy, cxy):
        if i < j and _overlaps([tuple(p) for p in cxy[i]], [tuple(p) for p in cxy[j]]):
            raise RegionalPatchRefusal("PATCH_OVERLAP", "kernel cells overlap")
    station_rows = set(station_row)
    for r in sorted({r for c in cells for r in c} - station_rows):
        p = tuple(pts[r])
        if not _strict_inside(p, outer_poly):
            raise RegionalPatchRefusal("PATCH_OUTSIDE_DOMAIN", "interior node outside the domain")
        for _, hxy in loops[1:]:
            if _strict_inside(p, [tuple(q) for q in hxy]):
                raise RegionalPatchRefusal("PATCH_FILLS_HOLE", "interior node lies inside a hole")
    domain_area = sum(_exact_double_area(xy) for _, xy in loops)
    cell_area = sum(_exact_double_area(c) for c in cxy)
    if cell_area != domain_area:
        raise RegionalPatchRefusal("PATCH_COVERAGE", "cell areas do not cover the prescribed material domain")
    for i, j in _bbox_overlap_pairs(cxy, retained_xy):
        if _overlaps([tuple(p) for p in cxy[i]], [tuple(p) for p in retained_xy[j]]):
            raise RegionalPatchRefusal("PATCH_OVERLAPS_RETAINED", "new domain overlaps retained cells")

    # Node ids: shared interface stations keep ids; the rest are fresh, sorted.
    row_node: dict[int, int] = {}
    new_nodes: dict[int, tuple[float, float, float]] = {}
    new_uv: dict[int, tuple[float, float]] = {}
    node_xy: dict[int, np.ndarray] = {}
    renumbered: list[tuple[int, int]] = []
    station_of_row = {r: st for (st, _), r in zip(stations, station_row)}
    for (st, p), r in zip(stations, station_row):
        if st.node_id is not None and st.node_id in shared:
            row_node[r] = st.node_id
            node_xy[st.node_id] = p
    used_rows = sorted({r for c in cells for r in c})
    pending = sorted((r for r in used_rows if r not in row_node), key=lambda r: (pts[r][1], pts[r][0]))
    for r in pending:
        nid = ids.next_node()
        row_node[r] = nid
        st = station_of_row.get(r)
        if st is not None and st.node_id is not None:
            renumbered.append((st.node_id, nid))
        new_nodes[nid] = tuple(float(v) for v in (st.xyz if st is not None else chart.to3(pts[r])))
        new_uv[nid] = (float(pts[r][0]), float(pts[r][1]))
        node_xy[nid] = pts[r]
    final = []
    for cell in cells:
        c = [row_node[r] for r in cell]
        k = c.index(min(c))
        final.append(tuple(c[k:] + c[:k]))
    final.sort()
    bkeys = {_ekey(row_node[a], row_node[b]) for a, b in expected}

    mid_id: dict[tuple[int, int], int] = {}
    if quadratic:
        explicit: dict[tuple[int, int], tuple[float, float, float]] = {}
        old_pos: dict[tuple[int, int], tuple[float, float, float]] = {}
        old_mid_id: dict[tuple[int, int], int] = {}
        curved: set[tuple[int, int]] = set()
        shared_mid: dict[tuple[int, int], int] = {}
        for (lp, _), base in zip(loops, bases):
            n = len(lp.stations)
            for i in range(n):
                ra = row_node[station_row[base + i]]
                rb = row_node[station_row[base + (i + 1) % n]]
                key = _ekey(ra, rb)
                oa, ob = lp.stations[i].node_id, lp.stations[(i + 1) % n].node_id
                if oa is not None and ob is not None:
                    okey = _ekey(oa, ob)
                    if okey in ipairs:
                        if okey not in old_mid:
                            raise RegionalPatchRefusal("MISSING_MIDSIDE", f"interface edge {okey} has no midside")
                        shared_mid[key] = int(old_mid[okey])
                    elif okey in old_mid:
                        old_pos[key] = tuple(float(v) for v in mesh.nodes[old_mid[okey]])
                        old_mid_id[key] = int(old_mid[okey])
                if lp.segment_midsides is not None and lp.segment_midsides[i] is not None:
                    explicit[key] = tuple(float(v) for v in lp.segment_midsides[i])
                elif not lp.straight:
                    curved.add(key)

        def pos(nid: int):
            return np.asarray(new_nodes[nid] if nid in new_nodes else mesh.nodes[nid], dtype=float)

        def mid(a: int, b: int) -> int:
            key = _ekey(a, b)
            if key in mid_id:
                return mid_id[key]
            if key in shared_mid:
                mid_id[key] = shared_mid[key]
                return mid_id[key]
            if key in curved and key not in explicit:
                raise RegionalPatchRefusal("CURVED_BOUNDARY_MIDSIDE_REQUIRED", f"edge {key}")
            nid = ids.next_node()
            if key in explicit:
                xyz = explicit[key]
            elif key in old_pos:
                xyz = old_pos[key]
            else:
                xyz = tuple(float(v) for v in 0.5 * (pos(a) + pos(b)))
            new_nodes[nid] = xyz
            mid_id[key] = nid
            if key in old_mid_id and key not in explicit:
                renumbered.append((old_mid_id[key], nid))
            return nid

        final = [tuple(c) + tuple(mid(c[k], c[(k + 1) % len(c)]) for k in range(len(c))) for c in final]

    hole_edges: dict[int, tuple[list[int], set[tuple[int, int]]]] = {}
    for edge_id, idx in hole_edge_loops.items():
        lp = loops[idx][0]
        n, base = len(lp.stations), bases[idx]
        seq, keys = [], set()
        for i in range(n):
            a = row_node[station_row[base + i]]
            b = row_node[station_row[base + (i + 1) % n]]
            seq.append(a)
            if quadratic:
                seq.append(mid_id[_ekey(a, b)])
            keys.add(_ekey(a, b))
        hole_edges[edge_id] = (seq, keys)
    return _Patch(comp, new_nodes, new_uv, final, bkeys, total, renumbered, hole_edges)


# ------------------------------------------------------------ stage + assemble


def _stage_replacement(mesh: Mesh, base: Mesh, face_id: int, patches: list[_Patch], shared: set[int],
                       affected: set[int], removed_nodes: set[int], ids: _Ids, request):
    """Replacement mesh (permanent ids already) carrying every rebuilt association.

    Free-boundary nodes that are re-created at the same XYZ under fresh ids are
    substituted in the vertex/edge/member node lists (moved from ``base`` to the
    replacement so the owner assembler accepts the rebuilt entry).  A retired
    associated node that is *not* such a re-creation is refused unless the owner
    explicitly retires the whole geometry key.
    """

    rmap = {old: new for p in patches for old, new in p.renumbered}
    post: dict[tuple[str, int], list[int]] = {}
    retired_keys = request.retired_geometry_keys or {}
    rep = Mesh(geometry_model_id=mesh.geometry_model_id, geometry_revision=mesh.geometry_revision,
               order=mesh.order)
    for n in sorted(shared):
        rep.nodes[n] = np.array(mesh.nodes[n], dtype=float)
    new_by_patch: list[list[int]] = []
    for p in patches:
        for nid, xyz in p.new_nodes.items():
            rep.nodes[nid] = np.array(xyz, dtype=float)
        mine = []
        for c in p.cells:
            eid = ids.next_elem()
            mine.append(eid)
            (rep.quads if len(c) in (4, 8) else rep.tris)[eid] = tuple(c)
        new_by_patch.append(mine)
    all_new = [e for m in new_by_patch for e in m]
    rep.elements_of_face[face_id] = list(all_new)

    for key, olds in mesh.elements_of_face.items():
        if key != face_id and affected & set(olds):
            raise RegionalPatchRefusal("SHARED_ELEMENT_FACE", f"replaced elements also belong to face {key}")

    retired_edges = {int(k) for k in retired_keys.get("edge", ())}
    for vid, n in mesh.node_of_vertex.items():
        if n in removed_nodes and vid not in {int(k) for k in retired_keys.get("vertex", ())}:
            if n not in rmap:
                raise RegionalPatchRefusal("ASSOCIATION_UPDATE_REQUIRED", f"vertex {vid} node {n} is dropped")
            rep.node_of_vertex[vid] = rmap[n]
    for name, kind in (("nodes_of_edge", "edge"), ("offset_nodes_of_edge", "edge"), ("nodes_of_member", "member")):
        gone = {int(k) for k in retired_keys.get(kind, ())}
        for key, seq in getattr(mesh, name).items():
            if key in gone or not any(n in removed_nodes for n in seq):
                continue
            if any(n in removed_nodes and n not in rmap for n in seq):
                raise RegionalPatchRefusal(
                    "ASSOCIATION_UPDATE_REQUIRED",
                    f"{name}[{key}] loses nodes the patch does not re-create; owner update required")
            getattr(base, name).pop(key)
            post[(name, key)] = [rmap.get(n, n) for n in seq]
    # Activity: new elements inherit one consistent value, otherwise refuse.
    for p, mine in zip(patches, new_by_patch):
        comp = set(p.comp)
        for name in ("elements_of_sheet", "elements_of_member"):
            table = getattr(rep, name)
            for key, olds in getattr(mesh, name).items():
                inter = comp & set(olds)
                if not inter:
                    continue
                if inter != comp:
                    raise RegionalPatchRefusal("MIXED_ASSOCIATION", f"{name}[{key}] covers part of a replaced component")
                table.setdefault(key, []).extend(mine)
        for key, olds in mesh.elements_of_edge.items():
            inter = comp & set(olds)
            if not inter or key in retired_edges:
                continue
            members = {rmap.get(n, n) for n in mesh.nodes_of_edge.get(key, ())}
            hit = []
            for eid, c in zip(mine, p.cells):
                cs = c[:4] if len(c) in (4, 8) else c[:3]
                if any(_ekey(cs[i], cs[(i + 1) % len(cs)]) in p.boundary_keys
                       and cs[i] in members and cs[(i + 1) % len(cs)] in members
                       for i in range(len(cs))):
                    hit.append(eid)
            if len(hit) != len(inter):
                raise RegionalPatchRefusal(
                    "EDGE_ASSOCIATION_INCOMPLETE",
                    f"edge {key}: {len(inter)} replaced elements but {len(hit)} rebuilt; owner update required")
            rep.elements_of_edge.setdefault(key, []).extend(hit)
        for edge_id, (seq, keys) in p.hole_edges.items():
            rep.nodes_of_edge[edge_id] = list(seq)
            rep.elements_of_edge[edge_id] = [
                eid for eid, c in zip(mine, p.cells)
                if any(_ekey((c[:4] if len(c) in (4, 8) else c[:3])[i],
                             (c[:4] if len(c) in (4, 8) else c[:3])[(i + 1) % (4 if len(c) in (4, 8) else 3)]) in keys
                       for i in range(4 if len(c) in (4, 8) else 3))]
        acts = {e: mesh.activity[e] for e in comp if e in mesh.activity}
        if acts:
            if len(acts) != len(comp) or len(set(acts.values())) != 1:
                raise RegionalPatchRefusal("MIXED_ACTIVITY", "replaced elements carry differing activity")
            value = next(iter(acts.values()))
            for eid in mine:
                rep.activity[eid] = value
    return rep, post


def _classify(message: str) -> str:
    low = message.lower()
    if any(w in low for w in ("aspect ratio", "warp", "scaled jacobian")):
        return "QUALITY_REJECTED"
    if "conform" in low or "non-manifold" in low:
        return "OWNER_CONFORMITY_REJECTED"
    if "interface" in low or "stations" in low:
        return "INTERFACE_NOT_PRESERVED"
    if any(w in low for w in ("diagnostic", "schema", "hybrid_diagnostics", "structural_preparation", "metadata")):
        return "METADATA_REFUSED"
    if any(w in low for w in ("loses", "rebuild", "retired", "spans", "association")):
        return "ASSOCIATION_UPDATE_REQUIRED"
    return "OWNER_ASSEMBLY_REJECTED"


# --------------------------------------------------------------- verification


def verify_regional_patch(old: Mesh, new: Mesh, face_id: int, replaced: Sequence[int],
                          new_ids: Sequence[int], *, require_all_quad: bool = True) -> tuple[str, ...]:
    """Re-prove the published patch from the meshes alone; raises on failure."""

    def fail(msg: str):
        raise RegionalPatchRefusal("VERIFICATION_FAILED", msg)

    try:
        _validate_structure(new, "candidate")
    except MeshError as exc:
        fail(f"candidate structure/associations: {exc}")
    old_s, new_s = _shells(old), _shells(new)
    replaced, new_ids = set(replaced), set(new_ids)
    checks = []
    for eid, conn in old_s.items():
        if eid in replaced:
            if eid in new_s:
                fail(f"replaced element {eid} still present")
            continue
        if new_s.get(eid) != conn:
            fail(f"retained element {eid} changed")
        for n in conn:
            if not np.array_equal(old.nodes[n], new.nodes[n]):
                fail(f"retained node {n} moved")
    checks.append("retained_cells_exact")
    if set(new_s) != (set(old_s) - replaced) | new_ids or (new_ids & set(old_s)):
        fail("element id bookkeeping")
    expect_face = (set(old.elements_of_face.get(face_id, ())) - replaced) | new_ids
    if set(new.elements_of_face.get(face_id, ())) != expect_face:
        fail("elements_of_face provenance")
    checks.append("face_provenance")
    want = (8, 6) if old.is_quadratic else (4, 3)
    for eid in new_ids:
        conn = new_s[eid]
        if len(conn) not in want or len(set(conn)) != len(conn) or any(n not in new.nodes for n in conn):
            fail(f"element {eid} order/nodes")
        if require_all_quad and len(conn) not in (4, 8):
            fail(f"element {eid} is not a quad")
    checks.append("complete_order_nodes")
    directed: dict[tuple[int, int], int] = {}
    mids: dict[tuple[int, int], int] = {}
    for eid in new.elements_of_face.get(face_id, ()):
        conn = new_s[eid]
        cs = _corner_ids(conn)
        for k in range(len(cs)):
            e = (cs[k], cs[(k + 1) % len(cs)])
            if e in directed:
                fail(f"edge {e} used twice in the same direction (overlap or flipped cell)")
            directed[e] = eid
            if len(conn) in (6, 8):
                key = _ekey(*e)
                if mids.setdefault(key, conn[len(cs) + k]) != conn[len(cs) + k]:
                    fail(f"edge {key} has two midside nodes")
    checks.append("face_edge_orientation_and_midsides")
    for eid, seq in old.nodes_of_edge.items():
        if eid in new.nodes_of_edge and [n for n in new.nodes_of_edge[eid] if n in old.nodes] \
                != [n for n in seq if n in new.nodes]:
            fail(f"seam/edge {eid} node identity changed")
    for vid, n in old.node_of_vertex.items():
        if n in new.nodes and new.node_of_vertex.get(vid) != n:
            fail(f"vertex {vid} identity changed")
    checks.append("seam_vertex_identity")
    return tuple(checks)


# ------------------------------------------------------------------ top level


def _check_scope(request: RegionalPatchRequest) -> None:
    if request.held_case is not None and request.held_case in HELD_CASES:
        raise RegionalPatchRefusal("OWNER_HOLD", f"{request.held_case} is held by the owner", held_case=request.held_case)
    kind = request.surface_kind
    if kind == "plane":
        return
    if kind in _CURVED_FAMILIES:
        raise RegionalPatchRefusal(
            "CURVED_FAMILY_UNIMPLEMENTED",
            f"no owner-proven chart and source correspondence exists for {kind!r}; full-face remesh required",
            family=kind)
    raise RegionalPatchRefusal("UNKNOWN_SURFACE_KIND", f"{kind!r} is not a known surface family")


def _check_prerequisites(request: RegionalPatchRequest) -> None:
    mesh = request.old_mesh
    try:
        _validate_structure(mesh, "old mesh")
    except MeshError as exc:
        raise RegionalPatchRefusal("INVALID_MESH", str(exc)) from exc
    if mesh.geometry_model_id is None or mesh.geometry_revision is None:
        raise RegionalPatchRefusal("GEOMETRY_PROVENANCE_MISSING", "mesh has no geometry model id/revision")
    if not isinstance(request.quality, RegionalQualityLimits):
        raise RegionalPatchRefusal("QUALITY_LIMITS_REQUIRED", "explicit RegionalQualityLimits are required")
    if request.source_check is None:
        raise RegionalPatchRefusal("SOURCE_PROOF_REQUIRED", "a source correspondence check is required")
    if request.tolerances.planarity is None or not request.tolerances.planarity >= 0.0:
        raise RegionalPatchRefusal("PLANARITY_TOLERANCE_REQUIRED", "owner planarity tolerance is required")
    holes = list(request.added_holes)
    edges = [int(e) for e in request.added_hole_edge_ids]
    if len(edges) != len(holes) or len(set(edges)) != len(edges):
        raise RegionalPatchRefusal("NEW_HOLE_ASSOCIATIONS_REQUIRED",
                                   "each added hole needs its own new geometry edge id")
    for e in edges:
        if e in mesh.nodes_of_edge or e in mesh.elements_of_edge or e in mesh.offset_nodes_of_edge:
            raise RegionalPatchRefusal("NEW_HOLE_ASSOCIATIONS_REQUIRED", f"edge id {e} already exists")


def _project_sizing_zones(request, chart):
    """Validate convex physical footprints before they can authorize a sized result."""
    result = []
    try:
        for vertices, wanted in request.sizing_zones:
            if isinstance(wanted, (str, bool, np.bool_)) or not math.isfinite(wanted) or wanted <= 0:
                raise ValueError("edge target must be finite and positive")
            xyz = np.asarray(vertices, dtype=float)
            if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or not np.isfinite(xyz).all():
                raise ValueError("expected a finite polygon of three-dimensional points")
            if any(abs(chart.height(p)) > request.tolerances.planarity for p in xyz):
                raise ValueError("polygon is off the patch plane")
            polygon = np.array([chart.to2(p) for p in xyz])
            if len(set(map(tuple, polygon))) != len(polygon) or not _exact_double_area(polygon):
                raise ValueError("polygon is degenerate")
            # Every vertex lies in the same closed half-plane of each oriented
            # edge. This also rules out self-crossing polygons of positive area.
            if _exact_double_area(polygon) < 0:
                polygon = polygon[::-1]
            if any(_orient(a, b, p) < 0 for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
                   for p in polygon):
                raise ValueError("polygon must be a simple convex sizing footprint")
            result.append((polygon, float(wanted)))
    except (TypeError, ValueError, OverflowError) as error:
        raise RegionalPatchRefusal("BAD_SIZING_ZONE", str(error)) from error
    return tuple(result)


def _check_sizing_zones(mesh, chart, ids, zones):
    """Test physical edge lengths of actual new cells with positive footprint overlap."""
    for polygon, target in zones:
        for eid in ids:
            if eid not in mesh.quads:
                continue
            xyz = np.asarray([mesh.nodes[n] for n in mesh.quads[eid][:4]])
            if np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1).max() > target:
                continue
            if _overlaps([chart.to2(point) for point in xyz], polygon):
                break
        else:
            raise RegionalPatchRefusal("LOCAL_SIZE_UNMET",
                                       f"no new quad overlapping the sizing zone meets edge target {target:g}")


def remesh_region(request: RegionalPatchRequest) -> RegionalPatchResult:
    """Locally remesh ``request``'s region; raise :class:`RegionalPatchRefusal`."""

    _check_scope(request)
    _check_prerequisites(request)
    mesh = request.old_mesh
    shells = _shells(mesh)
    face_list = [int(e) for e in mesh.elements_of_face.get(request.face_id, ())]
    if not face_list or any(e not in shells for e in face_list):
        raise RegionalPatchRefusal("NO_FACE_PROVENANCE", f"face {request.face_id} has no usable element provenance")
    face_set = set(face_list)
    if any(len(shells[e]) not in ((6, 8) if mesh.is_quadratic else (3, 4)) for e in face_list):
        raise RegionalPatchRefusal("MIXED_ORDER", "face elements do not match mesh order")
    chart = _plane_chart(mesh, min(face_set))
    sizing_zones = _project_sizing_zones(request, chart)
    tol = request.tolerances.planarity
    for e in face_list:
        cell = [chart.to2(mesh.nodes[n]) for n in shells[e]]
        for n in shells[e]:
            if abs(chart.height(mesh.nodes[n])) > tol:
                raise RegionalPatchRefusal("NOT_PLANAR", f"node {n} is off the face plane")
        if not _area2(cell[:len(_corner_ids(shells[e]))]) > 0.0:
            raise RegionalPatchRefusal("MIXED_ORIENTATION", f"element {e} is not counter-clockwise in the face frame")
    kernel = request.kernel or owner_planar_kernel
    route_name = "planar_owner_kernel" if request.kernel is None else "planar_injected_kernel"
    grids = tuple(k for k in (request.face_id,) if k in mesh.grid_of_face or k in mesh.block_grids_of_face)
    if grids and not request.remove_mapped_grid:
        raise RegionalPatchRefusal("MAPPED_GRID_REMOVAL_REQUIRED",
                                   "the face has a mapped grid that a local patch invalidates; set remove_mapped_grid")
    users = _edge_users(shells)
    old_mid = _old_mids(shells)
    region = _select_region(request, chart, shells, face_set)
    attempts: list[dict[str, Any]] = []
    growable = request.new_domain is None and request.domain_builder is None
    for ring in range(max(0, int(request.max_growth_rings)) + 1):
        try:
            return _attempt(request, mesh, shells, users, old_mid, chart, kernel, route_name,
                            mesh.is_quadratic, region, ring, attempts, face_set, grids, sizing_zones)
        except RegionalPatchRefusal as refusal:
            attempts.append({"ring": ring, "code": refusal.code, "elements": len(region)})
            refusal.details["attempts"] = tuple(attempts)
            if refusal.code not in _GROWABLE or not growable or ring >= request.max_growth_rings:
                raise
            nodes = {n for e in region for n in shells[e]}
            grown = sorted(set(region) | {e for e in face_set if nodes & set(shells[e])})
            if len(grown) == len(region) or len(grown) >= len(face_set):
                refusal.details["growth_stopped"] = "LOCAL_REGION_EXHAUSTED"
                raise
            region = grown
    raise RegionalPatchRefusal("INTERNAL", "unreachable")  # pragma: no cover


def _attempt(request, mesh, shells, users, old_mid, chart, kernel, route_name, quadratic,
             region, ring, attempts, face_set, grids, sizing_zones) -> RegionalPatchResult:
    comps = _components(shells, region)
    if request.new_domain is not None and len(comps) != 1:
        raise RegionalPatchRefusal("EXPLICIT_DOMAIN_NEEDS_ONE_REGION", f"{len(comps)} components")
    ids = _Ids(mesh, request)
    boundaries = [_region_boundary(mesh, shells, users, chart, c, i) for i, c in enumerate(comps)]
    affected = set(region)
    conn_all = _element_nodes(mesh)
    aff_nodes = set().union(*(conn_all[e] for e in affected))
    out_nodes = set().union(*(conn_all[e] for e in conn_all if e not in affected)) if len(conn_all) > len(affected) else set()
    shared = aff_nodes & out_nodes
    retained_xy = [np.array([chart.to2(mesh.nodes[n]) for n in _corner_ids(shells[e])])
                   for e in sorted(face_set - affected)]

    drop = {int(n) for n in request.drop_hole_node_ids}
    dropped_found: set[int] = set()
    hole_edge_of_loop: dict[int, int] = {}
    assigned: dict[int, list[PatchLoop]] = {b.component: [] for b in boundaries}
    for hole, edge_id in zip(request.added_holes, request.added_hole_edge_ids):
        probe = tuple(chart.to2(hole.stations[0].xyz))
        owners = [b for b in boundaries if _strict_inside(
            probe, [tuple(chart.to2(s.xyz)) for s in b.outer.stations])]
        if len(owners) != 1:
            raise RegionalPatchRefusal("ADDED_HOLE_OUTSIDE_REGION", "hole is not inside exactly one replaced region")
        assigned[owners[0].component].append(hole)
        hole_edge_of_loop[id(hole)] = int(edge_id)
    patches, size_used = [], None
    for b, comp in zip(boundaries, comps):
        if request.new_domain is not None:
            domain = request.new_domain
        elif request.domain_builder is not None:
            domain = request.domain_builder(b)
        else:
            keep = []
            for h in b.holes:
                hit = {s.node_id for s in h.stations} & drop
                if hit:
                    if any(h.interface):
                        raise RegionalPatchRefusal("DROP_HOLE_NOT_FREE", "hole loop touches retained cells")
                    dropped_found |= hit
                else:
                    keep.append(h)
            domain = NewDomain(b.outer, tuple(keep) + tuple(assigned[b.component]))
        if request.target_size is not None:
            size = float(request.target_size)
        else:
            lens = [float(np.linalg.norm(np.asarray(mesh.nodes[a]) - np.asarray(mesh.nodes[c])))
                    for lp in (b.outer, *b.holes) for a, c in zip(
                        [s.node_id for s in lp.stations],
                        [s.node_id for s in lp.stations[1:]] + [lp.stations[0].node_id])]
            size = float(sum(lens) / len(lens))
        if not (size > 0.0 and math.isfinite(size)):
            raise RegionalPatchRefusal("BAD_TARGET_SIZE", "target size must be positive and finite")
        size_used = size
        patches.append(_build_patch(request, mesh, users, old_mid, chart, comp, b, domain, shared,
                                    retained_xy, ids, size, kernel, quadratic, hole_edge_of_loop, sizing_zones))
    if drop - dropped_found and request.new_domain is None and request.domain_builder is None:
        raise RegionalPatchRefusal("DROP_HOLE_NOT_FOUND", f"nodes {sorted(drop - dropped_found)} are on no free hole loop")

    # Couplings that interpolate inside a replaced element cannot keep their weights.
    for cid, cp in mesh.couplings.items():
        pl = set(int(n) for n in cp.plate_nodes)
        if len(pl) > 1 and any(pl <= set(shells[e]) for e in affected):
            raise RegionalPatchRefusal("COUPLING_INTERPOLATION_REPLACED", f"coupling {cid} interpolates in a replaced element")
    removed_nodes = aff_nodes - out_nodes
    for pair in mesh.declared_plate_junction_edges:
        if set(pair) & removed_nodes:
            raise RegionalPatchRefusal("JUNCTION_EDGE_REPLACED", f"declared junction edge {tuple(pair)} touches a replaced node")

    base = copy.deepcopy(mesh)
    for k in grids:
        base.grid_of_face.pop(k, None)
        base.block_grids_of_face.pop(k, None)
    rep, post = _stage_replacement(mesh, base, request.face_id, patches, shared, affected,
                             removed_nodes, ids, request)
    fresh = [n for n in rep.nodes if n not in shared]
    try:
        asm = assemble_regional_replacement(
            base, rep, quality=request.quality,
            node_ids={n: n for n in fresh}, element_ids={e: e for e in _element_nodes(rep)},
            affected_element_ids=affected, interface_node_ids={n: n for n in shared},
            stations={n: mesh.nodes[n] for n in shared}, coordinate_tolerance=0.0,
            discard_stale_diagnostics=request.discard_stale_diagnostics,
            retired_geometry_keys=request.retired_geometry_keys)
    except RegionalPatchRefusal:
        raise
    except MeshError as exc:
        raise RegionalPatchRefusal(_classify(str(exc)), str(exc)) from exc
    new = asm.mesh
    for (name, key), seq in post.items():
        getattr(new, name)[key] = seq
    new_ids = list(asm.new_element_ids)

    checks = list(verify_regional_patch(mesh, new, request.face_id, affected, new_ids,
                                        require_all_quad=request.require_all_quad))
    if sizing_zones:
        _check_sizing_zones(new, chart, new_ids, sizing_zones)
        checks.append("new_quad_edge_target_in_each_sizing_zone")
    checks += ["owner_assembly_conformity_and_explicit_quality", "candidate_associations_valid",
               "coverage_boundary_equals_domain", "interior_overlap_hole_and_area_coverage",
               "no_overlap_with_retained_cells", "strict_positive_convex_cells",
               "interface_directed_segments_exact", "fixed_stations_reproduced_exactly",
               "planarity_within_owner_tolerance", "monotonic_ids_above_high_water"]
    if request.require_all_quad:
        checks.append("all_quad_boundary_parity")
    unchecked = ["source_geometry_projection_beyond_caller_source_check"]
    if request.added_holes:
        unchecked.append("new_hole_edge_closed_loop_node_list_convention_and_seeding")
    if request.kernel is not None:
        unchecked.append("owner_kernel_not_used_injected_kernel")
    if request.quality_gate is not None and not request.quality_gate(new, tuple(new_ids)):
        raise RegionalPatchRefusal("QUALITY_REJECTED", "caller quality gate rejected the patch")
    renumbered = tuple(sorted(r for p in patches for r in p.renumbered))
    report = RegionalPatchReport(
        status="accepted", face_id=request.face_id, route=route_name,
        order="quadratic" if quadratic else "linear", kernel_injected=request.kernel is not None,
        replaced_element_ids=tuple(sorted(affected)), new_element_ids=tuple(sorted(new_ids)),
        new_node_ids=tuple(asm.new_node_ids), removed_node_ids=tuple(asm.retired_node_ids),
        retained_node_ids=tuple(asm.retained_node_ids), retained_element_ids=tuple(asm.retained_element_ids),
        interface_node_ids=tuple(asm.interface_node_ids), renumbered_free_nodes=renumbered,
        dropped_geometry_nodes={k: tuple(v) for k, v in asm.dropped_geometry_keys.items()},
        removed_grids=tuple(grids),
        new_hole_edges={e: tuple(s) for p in patches for e, (s, _) in p.hole_edges.items()},
        discarded_diagnostic_keys=tuple(asm.discarded_diagnostic_keys),
        node_high_water=int(ids.node_hw), element_high_water=int(ids.elem_hw),
        quality=None if asm.quality is None else asm.quality.as_dict(),
        retained_element_count=len(face_set) - len(affected), components=len(comps),
        boundary_station_count=sum(p.stations for p in patches),
        all_quad=all(len(c) in (4, 8) for p in patches for c in p.cells),
        growth_rings=ring, attempts=tuple(attempts), checks=tuple(checks),
        unchecked=tuple(unchecked), target_size=float(size_used), provenance=dict(request.provenance),
    )
    if not request.source_check(mesh, new, report):
        raise RegionalPatchRefusal("SOURCE_CORRESPONDENCE_REJECTED", "caller source check rejected the patch")
    all_uv = {n: uv for p in patches for n, uv in p.new_uv.items()}
    return RegionalPatchResult(new, report, all_uv)


def try_remesh_region(request: RegionalPatchRequest) -> RegionalPatchResult | RegionalPatchRefusal:
    """Non-raising form: returns the typed refusal so the caller can fall back."""

    try:
        return remesh_region(request)
    except RegionalPatchRefusal as refusal:
        return refusal
