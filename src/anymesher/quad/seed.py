"""Target-size-driven constrained T3 seed for the planar quad advancing front."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import ceil, floor
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
import numpy as np
from anygeometry.model import GeometryModel
from ..errors import MeshError
from ..refinement import SizeField
from ..triangulation import PlanarTriangulation, orient2d, triangulate_polygon
from .boundary import BoundaryStationKey, BoundaryStationRegistry
from .domain import CylindricalQuadDomain, PlanarQuadDomain, Vec2
from .state import QuadMeshState

def _shoelace(loop_points: np.ndarray) -> float:
    loop = np.asarray(loop_points, dtype=float)
    if loop.ndim != 2 or loop.shape[1] < 2 or len(loop) < 3:
        raise MeshError("shoelace loop must contain at least three points")
    x = loop[:, 0]
    y = loop[:, 1]
    value = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if not np.isfinite(value):
        raise MeshError("shoelace area is not finite")
    return value


@dataclass(frozen=True)
class PlanarQuadSeed:
    domain: PlanarQuadDomain
    registry: BoundaryStationRegistry
    target_size: float
    boundary_keys: tuple[BoundaryStationKey, ...]
    triangulation: PlanarTriangulation
    state: QuadMeshState
    station_to_node: Mapping[BoundaryStationKey, int]
    targeted_reseed_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "station_to_node", MappingProxyType(dict(self.station_to_node)))

    @property
    def discrete_area(self) -> float:
        points = np.asarray(self.triangulation.points, dtype=float)
        area = _shoelace(points[np.asarray(self.triangulation.outer_loop, dtype=np.int64)])
        for ring in self.triangulation.hole_loops:
            area -= _shoelace(points[np.asarray(ring, dtype=np.int64)])
        if not np.isfinite(area) or area <= 0.0:
            raise MeshError("discrete meshed-domain area must be positive and finite")
        return area


# Seeds stay on the deterministic Python reference triangulation.  The
# compiled triangulator breaks exactly cocircular ties (every rectangular
# lattice cell) differently, so its seeds are valid but not byte-identical
# outside the planar PQ4b parity corpus (e.g. the CH9 h=0.3 cone), which would
# make quad-first output depend on how ANYmesher was installed.
_SEED_TRIANGULATION_BACKEND = "python"

# Lattice points closer than this fraction of the local size to a boundary
# segment leave a thin strip that the front can only close with slivers.
_LATTICE_BOUNDARY_CLEARANCE = 0.3


class _SegmentGrid:
    """Uniform-grid bucket of chart segments for bounded distance queries."""

    def __init__(self, rings: Sequence[Sequence[Vec2]], cell: float) -> None:
        self.cell = float(cell)
        self.buckets: dict[tuple[int, int], list[tuple[Vec2, Vec2]]] = {}
        for ring in rings:
            count = len(ring)
            for index in range(count):
                a = (float(ring[index][0]), float(ring[index][1]))
                b = (float(ring[(index + 1) % count][0]), float(ring[(index + 1) % count][1]))
                i0, j0 = self._key((min(a[0], b[0]) - cell, min(a[1], b[1]) - cell))
                i1, j1 = self._key((max(a[0], b[0]) + cell, max(a[1], b[1]) + cell))
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        self.buckets.setdefault((i, j), []).append((a, b))

    def _key(self, point: Sequence[float]) -> tuple[int, int]:
        return floor(point[0] / self.cell), floor(point[1] / self.cell)

    def closer_than(self, point: Sequence[float], distance: float) -> bool:
        """True when some segment is strictly closer than ``distance``.

        A segment within ``distance`` has its closest point inside the query
        square, and every segment is bucketed in each cell its bounding box
        touches, so scanning the cells covering that square is exact for any
        ``distance``.
        """
        x, y = float(point[0]), float(point[1])
        limit = distance * distance
        i0, j0 = self._key((x - distance, y - distance))
        i1, j1 = self._key((x + distance, y + distance))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for a, b in self.buckets.get((i, j), ()):
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    length2 = dx * dx + dy * dy
                    t = 0.0 if length2 <= 0.0 else max(
                        0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / length2)
                    )
                    ex, ey = x - (a[0] + t * dx), y - (a[1] + t * dy)
                    if ex * ex + ey * ey < limit:
                        return True
        return False


class _Buckets:
    """Uniform-grid buckets of items with axis-aligned bounds (2D or 3D).

    ``near(center, radius)`` yields every item whose bounds meet the query box,
    so any item within ``radius`` of ``center`` is returned (possibly with
    others and duplicates).  Callers apply their exact predicate to the result,
    which keeps accepted seeds identical to a full scan.
    """

    def __init__(self, cell: float) -> None:
        if not (np.isfinite(cell) and cell > 0.0):
            raise MeshError("bucket cell size must be positive and finite")
        self.cell = float(cell)
        self.items: dict[tuple[int, ...], list[int]] = {}

    def _range(self, lower: Sequence[float], upper: Sequence[float]):
        low = [floor(float(v) / self.cell) for v in lower]
        high = [floor(float(v) / self.cell) for v in upper]
        return product(*(range(a, b + 1) for a, b in zip(low, high)))

    def insert(self, item: int, lower: Sequence[float], upper: Sequence[float]) -> None:
        for key in self._range(lower, upper):
            self.items.setdefault(key, []).append(item)

    def near(self, center: Sequence[float], radius: float):
        lower = [float(v) - radius for v in center]
        upper = [float(v) + radius for v in center]
        for key in self._range(lower, upper):
            yield from self.items.get(key, ())


def _on_segment(p: Vec2, a: Vec2, b: Vec2, tol: float) -> bool:
    if p[0] < min(a[0], b[0]) - tol or p[0] > max(a[0], b[0]) + tol:
        return False
    if p[1] < min(a[1], b[1]) - tol or p[1] > max(a[1], b[1]) + tol:
        return False
    return abs((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) <= tol


def _strict_inside(p: Vec2, polygon: Sequence[Vec2], tol: float) -> bool:
    inside = False
    x, y = p
    for i, a in enumerate(polygon):
        b = polygon[(i+1) % len(polygon)]
        if _on_segment(p, a, b, tol):
            return False
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if crossing > x:
                inside = not inside
    return inside


def _strict_inside_mask(xs: np.ndarray, ys: np.ndarray, polygon: Sequence[Vec2], tol: float) -> np.ndarray:
    """``_strict_inside`` for many points at once, decision-identical.

    A point is rejected when any edge passes ``_on_segment`` (the scalar loop
    returns early, so edge order does not matter) and is otherwise inside on
    odd crossing parity.  Every comparison and crossing abscissa uses the
    scalar version's float operations in the same order.
    """
    on_edge = np.zeros(xs.shape, dtype=bool)
    inside = np.zeros(xs.shape, dtype=bool)
    count = len(polygon)
    for i in range(count):
        a = polygon[i]
        b = polygon[(i+1) % count]
        ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        near = ((xs >= min(ax, bx) - tol) & (xs <= max(ax, bx) + tol)
                & (ys >= min(ay, by) - tol) & (ys <= max(ay, by) + tol))
        if near.any():
            on_edge[near] |= np.abs((bx-ax)*(ys[near]-ay) - (by-ay)*(xs[near]-ax)) <= tol
        straddle = (ay > ys) != (by > ys)
        if straddle.any():
            crossing = ax + (ys[straddle]-ay)*(bx-ax)/(by-ay)
            inside[straddle] ^= crossing > xs[straddle]
    return inside & ~on_edge


def _interior_lattice(
    domain: PlanarQuadDomain,
    outer: Sequence[Vec2],
    holes: Sequence[Sequence[Vec2]],
    target_size: float,
    size_field: SizeField | None = None,
    *,
    coarse_points: np.ndarray | None = None,
) -> np.ndarray:
    h = float(target_size)
    if not np.isfinite(h) or h <= 0.0:
        raise MeshError("target_size must be positive and finite")
    if coarse_points is not None and (size_field is None or size_field.is_uniform):
        return coarse_points
    if size_field is None or size_field.is_uniform:
        min_x = min(p[0] for p in outer); max_x = max(p[0] for p in outer)
        min_y = min(p[1] for p in outer); max_y = max(p[1] for p in outer)
        tol = 1.0e-10 * max(max_x-min_x, max_y-min_y, 1.0)
        points = _uniform_lattice_points(outer, list(holes), h, min_x, max_x, min_y, max_y, tol)
        return np.asarray(points, dtype=float).reshape((-1, 2))
    min_x = min(p[0] for p in outer); max_x = max(p[0] for p in outer)
    min_y = min(p[1] for p in outer); max_y = max(p[1] for p in outer)
    tol = 1.0e-10 * max(max_x-min_x, max_y-min_y, 1.0)
    coarse = (coarse_points if coarse_points is not None else
              _uniform_lattice_points(outer, list(holes), h, min_x, max_x, min_y, max_y, tol))
    h_min = min(float(target_size), *(float(zone.size) for zone in size_field.zones))
    nx = int(np.ceil((max_x - min_x) / h_min))
    ny = int(np.ceil((max_y - min_y) / h_min))
    if nx * ny > 250000:
        raise MeshError("graded refinement lattice exceeds fine candidate budget")
    accepted: list[np.ndarray] = [np.asarray(p, dtype=float) for p in coarse]
    accepted_grid = _Buckets(0.75 * h)
    for index, p in enumerate(accepted):
        accepted_grid.insert(index, p, p)
    fine_segments = (_SegmentGrid((outer, *holes), _LATTICE_BOUNDARY_CLEARANCE * h)
                     if _LATTICE_BOUNDARY_CLEARANCE > 0.0 else None)
    seen = {(float(p[0]), float(p[1])) for p in coarse}
    # Domain containment for the whole candidate grid at once; the loop
    # below still computes each coordinate with the same scalar arithmetic.
    grid_x = np.tile(min_x + (np.arange(nx, dtype=float) + 0.5) * h_min, ny)
    grid_y = np.repeat(min_y + (np.arange(ny, dtype=float) + 0.5) * h_min, nx)
    contained = _strict_inside_mask(grid_x, grid_y, outer, tol)
    for hole in holes:
        contained &= ~_strict_inside_mask(grid_x, grid_y, hole, tol)
    for j in range(ny):
        y = min_y + (j + 0.5) * h_min
        if y > max_y - tol:
            continue
        for i in range(nx):
            x = min_x + (i + 0.5) * h_min
            if x > max_x - tol:
                continue
            point = (float(x), float(y))
            if not contained[j * nx + i]:
                continue
            key = point
            if key in seen:
                continue
            h_local = float(size_field.size_at(np.asarray([domain.lift(point)], dtype=float))[0])
            if h_local >= 0.999999 * target_size:
                continue
            if fine_segments is not None and fine_segments.closer_than(
                point, _LATTICE_BOUNDARY_CLEARANCE * h_local,
            ):
                continue
            if coarse_points is not None and min(
                x-min_x, max_x-x, y-min_y, max_y-y,
            ) < 0.5 * h_local:
                # A centred fine sample needs half a local interval before
                # the source boundary; otherwise it creates a thin last strip.
                continue
            if any(float(np.linalg.norm(accepted[index] - np.asarray(point, dtype=float))) < 0.75 * h_local
                   for index in accepted_grid.near(point, 0.75 * h_local)):
                continue
            accepted_grid.insert(len(accepted), point, point)
            accepted.append(np.asarray(point, dtype=float))
            seen.add(key)
    return np.asarray(accepted, dtype=float).reshape((-1, 2))


def _cylindrical_station_lattice(domain, registry, outer, target_size):
    """Distribute a rectangular chart's coarse backbone over its full extent.

    Exact opposite owner stations take precedence. Graded boundaries can
    disagree, so their interior backbone uses complete-span divisions instead
    of an ``h`` march that creates a thin remainder strip.
    """
    if not isinstance(domain, CylindricalQuadDomain) or len(domain.edge_uses) != 4 or domain.hole_edge_uses:
        return None
    points = np.asarray(outer, dtype=float)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    ratios = (upper-lower) / target_size
    if not any(1e-12 < ratio-floor(ratio) <= 0.1 for ratio in ratios):
        # Keep established well-proportioned cylinder charts on their
        # previous seed route. The station backbone addresses only a narrow
        # final interval, including graded charts with unequal chains.
        return None
    tolerance = 1e-10 * max(1.0, float(np.max(upper-lower)))
    axes: list[list[np.ndarray]] = [[], []]
    for edge_id, forward in domain.edge_uses:
        chain = np.asarray([domain.project(station.position)
                            for station in registry.chain(edge_id, forward)], dtype=float)
        if np.max(chain[:, 1])-np.min(chain[:, 1]) <= tolerance:
            axis = 0
        elif np.max(chain[:, 0])-np.min(chain[:, 0]) <= tolerance:
            axis = 1
        else:
            return None
        axes[axis].append(np.sort(chain[:, axis]))
    if any(len(pair) != 2 or any(abs(chain[0]-lower[axis]) > tolerance
                                    or abs(chain[-1]-upper[axis]) > tolerance
                                    for chain in pair)
           for axis, pair in enumerate(axes)):
        return None
    axes_values = []
    for axis, pair in enumerate(axes):
        if (len(pair[0]) == len(pair[1])
                and np.allclose(pair[0], pair[1], atol=tolerance, rtol=0.0)):
            axes_values.append(0.5 * (pair[0] + pair[1]))
        else:
            divisions = max(1, ceil((upper[axis]-lower[axis]) / target_size - 1e-12))
            axes_values.append(np.linspace(lower[axis], upper[axis], divisions+1))
    xs, ys = axes_values
    return np.asarray([(float(x), float(y)) for y in ys[1:-1] for x in xs[1:-1]],
                      dtype=float).reshape((-1, 2))


def source_chart_axis_lengths(
    geometry: GeometryModel, domain: PlanarQuadDomain, *, shortest: bool = False,
) -> tuple[float, float]:
    """Physical axis lengths on opposite source edges.

    Long edges set grid density; the shorter edge governs whether a grid or
    an interior point has enough physical room on a tapered face.
    """
    values = np.linspace(0.0, 1.0, 65)

    def length(along_u: bool) -> float:
        lengths = []
        for fixed in (0.0, 1.0):
            xyz = np.asarray([
                geometry.face_point(domain.face_id, float(t), fixed)
                if along_u else geometry.face_point(domain.face_id, fixed, float(t))
                for t in values
            ])
            lengths.append(float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()))
        return min(lengths) if shortest else max(lengths)

    return length(True), length(False)


def _source_aligned_point_lattice(
    geometry: GeometryModel,
    domain: PlanarQuadDomain,
    outer: Sequence[Vec2],
    target_size: float,
    cut: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[Vec2, ...]]:
    """Seed a source-parametric grid through a declared interior point.

    This changes only interior PQ2 candidates. Boundary stations still come
    from the shared source-edge registry, and the regular PQ-M1 front consumes
    the resulting CDT seed.
    """
    if len(domain.edge_uses) != 4 or domain.hole_edge_uses:
        raise MeshError("source-aligned point seed requires a four-sided face without holes")
    u0, v0 = map(float, cut)
    if not (0.0 < u0 < 1.0 and 0.0 < v0 < 1.0):
        raise MeshError("source-aligned seed point must be strictly interior")

    lengths = source_chart_axis_lengths(geometry, domain)
    short_lengths = source_chart_axis_lengths(geometry, domain, shortest=True)
    if any(min(at, 1.0-at) * length < .25 * target_size
           for at, length in zip((u0, v0), short_lengths)):
        raise MeshError("source-aligned seed point is too close to a face boundary for the requested size")

    def split_axis(length: float, at: float) -> np.ndarray:
        left = max(2, ceil(length * at / target_size))
        right = max(2, ceil(length * (1 - at) / target_size))
        return np.r_[np.linspace(0., at, left + 1),
                     np.linspace(at, 1., right + 1)[1:]]

    us = split_axis(lengths[0], u0)
    vs = split_axis(lengths[1], v0)
    if len(us) * len(vs) > 250000:
        raise MeshError("source-aligned point seed exceeds candidate budget")
    bounds = (max(p[0] for p in outer) - min(p[0] for p in outer),
              max(p[1] for p in outer) - min(p[1] for p in outer))
    tol = 1.0e-10 * max(1.0, *bounds)
    points = []
    grid: dict[tuple[int, int], Vec2] = {}
    for j, v in enumerate(vs[1:-1], start=1):
        for i, u in enumerate(us[1:-1], start=1):
            xyz = geometry.face_point(domain.face_id, float(u), float(v))
            point = domain.project(xyz)
            if not _strict_inside(point, outer, tol):
                raise MeshError("source-aligned point seed leaves the meshed face")
            points.append(point)
            grid[i, j] = point
    ci = int(np.argmin(abs(us-u0)))
    cj = int(np.argmin(abs(vs-v0)))
    if abs(us[ci]-u0) > 1e-12 or abs(vs[cj]-v0) > 1e-12:
        raise MeshError("source-aligned point seed lost the declared point")
    central = tuple(grid[i, j] for j in range(cj-1, cj+2)
                    for i in range(ci-1, ci+2))
    constraints = []
    for j in range(cj-1, cj+2):
        for i in range(ci-1, ci+1):
            constraints.append(np.asarray((grid[i, j], grid[i+1, j]), dtype=float))
    for i in range(ci-1, ci+2):
        for j in range(cj-1, cj+1):
            constraints.append(np.asarray((grid[i, j], grid[i, j+1]), dtype=float))
    return (np.asarray(points, dtype=float).reshape((-1, 2)),
            tuple(constraints), central)


def _uniform_lattice_points(
    outer: Sequence[Vec2],
    holes: list[Sequence[Vec2]],
    h: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    tol: float,
) -> list[tuple[float, float]]:
    xs = np.arange(min_x+h, max_x-tol, h, dtype=float)
    ys = np.arange(min_y+h, max_y-tol, h, dtype=float)
    clearance = _LATTICE_BOUNDARY_CLEARANCE * h
    segments = _SegmentGrid((outer, *holes), clearance) if clearance > 0.0 else None
    points: list[tuple[float, float]] = []
    # Row-major (y outer, x inner), the order of the former nested loop.
    grid_x = np.tile(xs, len(ys))
    grid_y = np.repeat(ys, len(xs))
    keep = _strict_inside_mask(grid_x, grid_y, outer, tol)
    for hole in holes:
        keep &= ~_strict_inside_mask(grid_x, grid_y, hole, tol)
    for row in np.flatnonzero(keep).tolist():
        point = (float(grid_x[row]), float(grid_y[row]))
        if segments is not None and segments.closer_than(point, clearance):
            continue
        # Cartesian lattice coordinates are unique by construction; avoid an
        # unnecessary O(N^2) duplicate scan over previously accepted points.
        points.append(point)
    return points


def _adaptive_interior_points(
    domain: PlanarQuadDomain,
    outer: Sequence[Vec2],
    holes: Sequence[Sequence[Vec2]],
    target_size: float,
    size_field: SizeField | None,
    cancellation_check: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Deterministic, physical-distance seed for nonrectangular owner charts.

    The source boundary is already fixed by the shared station registry.  A
    complete-span coarse grid favours quads; a finer candidate grid fills only
    places where the physical metric or the requested size field needs it.
    Neither pass introduces boundary points or changes source geometry.
    """
    rings = (tuple(outer), *(tuple(hole) for hole in holes))
    boundary = [
        (np.asarray(domain.lift(a), dtype=float),
         np.asarray(domain.lift(ring[(i + 1) % len(ring)]), dtype=float))
        for ring in rings for i, a in enumerate(ring)
    ]
    lower = np.min(np.asarray(outer, dtype=float), axis=0)
    upper = np.max(np.asarray(outer, dtype=float), axis=0)
    extent = upper - lower
    tol = 1.0e-10 * max(1.0, float(np.max(extent)))
    h_min = min(
        [float(target_size), *(float(zone.size) for zone in size_field.zones)]
    ) if size_field is not None else float(target_size)
    fine_step = h_min / 2.0
    if not np.isfinite(fine_step) or fine_step <= 0.0:
        raise MeshError("adaptive seed size must be positive and finite")
    fine_counts = np.maximum(1, np.ceil(extent / fine_step).astype(int))
    if int(fine_counts[0]) * int(fine_counts[1]) > 250000:
        raise MeshError("adaptive seed exceeds fine candidate budget")
    accepted_uv: list[tuple[float, float]] = []
    accepted_xyz: list[tuple[np.ndarray, float]] = []
    boundary_clearance = 0.45 if isinstance(domain, PlanarQuadDomain) else 0.28
    # Physical-space buckets for the spacing and clearance checks (identical
    # decisions to a full scan; see _Buckets).
    accepted_grid = _Buckets(0.85 * float(target_size))
    boundary_grid = _Buckets(boundary_clearance * float(target_size))
    for index, (start, end) in enumerate(boundary):
        boundary_grid.insert(index, np.minimum(start, end), np.maximum(start, end))

    def offer(point: tuple[float, float], *, coarse: bool) -> None:
        # Domain containment was decided for the whole grid (``candidates``).
        try:
            xyz = np.asarray(domain.lift(point), dtype=float)
        except MeshError as exc:
            # A curved chart's polygon may enclose points outside the
            # qualified owner parameter patch. They are never seed material.
            if "outside its bound patch" not in str(exc):
                raise
            return
        h_local = (
            float(size_field.size_at(xyz.reshape((1, 3)))[0])
            if size_field is not None else float(target_size)
        )
        clearance = float("inf")
        for index in boundary_grid.near(xyz, boundary_clearance * h_local):
            start, end = boundary[index]
            direction = end - start
            length_squared = float(direction @ direction)
            at = (max(0.0, min(1.0, float((xyz-start) @ direction) / length_squared))
                  if length_squared > 0.0 else 0.0)
            clearance = min(clearance, float(np.linalg.norm(xyz - (start + at * direction))))
        if clearance < boundary_clearance * h_local:
            return
        separation = 0.85 if not coarse and h_local >= 0.999999 * target_size else 0.65
        if any(float(np.linalg.norm(xyz - accepted_xyz[index][0]))
               < separation * min(h_local, accepted_xyz[index][1])
               for index in accepted_grid.near(xyz, separation * h_local)):
            return
        accepted_grid.insert(len(accepted_xyz), xyz, xyz)
        accepted_uv.append(point)
        accepted_xyz.append((xyz, h_local))

    def candidates(counts) -> list[list[tuple[tuple[float, float], bool]]]:
        """Grid rows of ``(point, strictly inside the domain)``."""
        rows = [[(float(lower[0] + (i + .5) * extent[0] / counts[0]),
                  float(lower[1] + (j + .5) * extent[1] / counts[1]))
                 for i in range(int(counts[0]))]
                for j in range(int(counts[1]))]
        xs = np.asarray([point[0] for row in rows for point in row], dtype=float)
        ys = np.asarray([point[1] for row in rows for point in row], dtype=float)
        inside = _strict_inside_mask(xs, ys, outer, tol)
        for hole in holes:
            inside &= ~_strict_inside_mask(xs, ys, hole, tol)
        flags = iter(inside.tolist())
        return [[(point, next(flags)) for point in row] for row in rows]

    coarse_counts = np.maximum(1, np.ceil(extent / float(target_size)).astype(int))
    for row in candidates(coarse_counts):
        if cancellation_check is not None:
            cancellation_check("quad-first:adaptive-coarse-seed")
        for point, inside in row:
            if inside:
                offer(point, coarse=True)
    for row in candidates(fine_counts):
        if cancellation_check is not None:
            cancellation_check("quad-first:adaptive-fine-seed")
        for point, inside in row:
            if inside:
                offer(point, coarse=False)
    return np.asarray(accepted_uv, dtype=float).reshape((-1, 2))


def _targeted_reseed(
    domain: PlanarQuadDomain,
    triangulation: PlanarTriangulation,
    target_size: float,
    size_field: SizeField | None,
    cancellation_check: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Offer at most one bounded pass near poor seed triangles and neighbours."""
    uv = np.asarray(triangulation.points, dtype=float)
    triangles = np.asarray(triangulation.triangles, dtype=int)
    if not len(triangles):
        return np.empty((0, 2), dtype=float)
    xyz = np.asarray([domain.lift(point) for point in uv], dtype=float)
    incident: dict[tuple[int, int], list[int]] = {}
    defects: list[tuple[float, int]] = []
    for row, (a, b, c) in enumerate(triangles):
        if cancellation_check is not None and row % 256 == 0:
            cancellation_check("quad-first:adaptive-reseed-audit")
        p, q, r = xyz[[a, b, c]]
        edges = (float(np.linalg.norm(q-p)), float(np.linalg.norm(r-q)),
                 float(np.linalg.norm(p-r)))
        double_area = float(np.linalg.norm(np.cross(q-p, r-p)))
        aspect = max(edges) ** 2 / double_area if double_area > 0 else float("inf")
        if aspect > 12.0:
            defects.append((aspect, row))
        for u, v in ((a, b), (b, c), (c, a)):
            incident.setdefault(tuple(sorted((int(u), int(v)))), []).append(row)
    if not defects:
        return np.empty((0, 2), dtype=float)
    limit = min(64, max(1, int(np.ceil(0.1 * len(triangles)))))
    affected: set[int] = set()
    for _, row in sorted(defects, key=lambda item: (-item[0], item[1]))[:limit]:
        affected.add(row)
        a, b, c = triangles[row]
        for u, v in ((a, b), (b, c), (c, a)):
            affected.update(incident[tuple(sorted((int(u), int(v))))])
    made: list[np.ndarray] = []
    made_xyz: list[np.ndarray] = []
    for row in sorted(affected):
        if cancellation_check is not None:
            cancellation_check("quad-first:adaptive-reseed")
        point = np.mean(uv[triangles[row]], axis=0)
        try:
            position = np.asarray(domain.lift(point), dtype=float)
        except MeshError as exc:
            if "outside its bound patch" not in str(exc):
                raise
            continue
        h_local = (
            float(size_field.size_at(position.reshape((1, 3)))[0])
            if size_field is not None else float(target_size)
        )
        if min(float(np.linalg.norm(position - other)) for other in xyz) < .30 * h_local:
            continue
        if any(float(np.linalg.norm(position - other)) < .45 * h_local
               for other in made_xyz):
            continue
        made.append(point)
        made_xyz.append(position)
    return np.asarray(made, dtype=float).reshape((-1, 2))


def build_planar_quad_seed(
    geometry: GeometryModel,
    face_id: int,
    target_size: float,
    *,
    domain: PlanarQuadDomain | None = None,
    registry: BoundaryStationRegistry | None = None,
    size_field: SizeField | None = None,
    source_aligned_cut: tuple[float, float] | None = None,
    layout_policy: str = "existing",
    cancellation_check: Callable[[str], None] | None = None,
) -> PlanarQuadSeed:
    if layout_policy not in ("existing", "adaptive"):
        raise MeshError("layout_policy must be 'existing' or 'adaptive'")
    start_revision = int(geometry.revision)
    start_model_id = str(geometry.model_id)
    domain = domain or PlanarQuadDomain.from_geometry(geometry, face_id)
    if domain.face_id != int(face_id):
        raise MeshError("domain face does not match requested face")
    domain.assert_current(geometry)
    if size_field is not None and abs(float(size_field.target_size) - float(target_size)) > 1.0e-12 * max(1.0, abs(float(target_size))):
        raise MeshError("provided size field target_size differs from seed target_size")
    registry = registry or BoundaryStationRegistry.for_domain(geometry, domain, target_size, size_field=size_field)
    registry.assert_current(geometry)
    if abs(registry.target_size-float(target_size)) > 1.0e-15*max(1.0,abs(float(target_size))):
        raise MeshError("boundary registry target_size differs from seed target_size")
    stations = registry.outer_stations(domain)
    outer = tuple(domain.project(station.position) for station in stations)
    hole_polys: list[np.ndarray] = []
    for index in range(len(domain.hole_edge_uses)):
        hole_stations = registry.hole_stations(domain, index)
        hole_points = [domain.project(station.position) for station in hole_stations]
        if len(hole_points) < 3:
            raise MeshError("hole boundary has fewer than three stations")
        hole_polys.append(np.asarray(hole_points, dtype=float))
    constraints: tuple[np.ndarray, ...] = ()
    central_points: tuple[Vec2, ...] = ()
    station_lattice = _cylindrical_station_lattice(domain, registry, outer, float(target_size))
    if source_aligned_cut is not None:
        interior, constraints, central_points = _source_aligned_point_lattice(
            geometry, domain, outer, float(target_size), source_aligned_cut
        )
        if size_field is not None and not size_field.is_uniform:
            supplemental = _interior_lattice(
                domain, outer, tuple(hole_polys), float(target_size), size_field
            )
            accepted = [np.asarray(point) for point in interior]
            for point in supplemental:
                h_local = float(size_field.size_at(
                    np.asarray([domain.lift(point)], dtype=float)
                )[0])
                if h_local >= .999999 * target_size:
                    continue
                if any(np.linalg.norm(existing-point) < .75*h_local
                       for existing in accepted):
                    continue
                accepted.append(np.asarray(point))
            interior = np.asarray(accepted, dtype=float).reshape((-1, 2))
    else:
        if layout_policy == "adaptive":
            interior = _adaptive_interior_points(
                domain, outer, tuple(hole_polys), float(target_size), size_field,
                cancellation_check,
            )
        else:
            interior = _interior_lattice(domain, outer, tuple(hole_polys), float(target_size),
                                         size_field, coarse_points=station_lattice)
    triangulation = triangulate_polygon(
        np.asarray(outer, dtype=float),
        holes=hole_polys,
        constraints=constraints,
        interior_points=interior,
        backend=_SEED_TRIANGULATION_BACKEND,
        cancellation_check=cancellation_check,
    )
    targeted_reseed_count = 0
    if layout_policy == "adaptive" and source_aligned_cut is None:
        reseed = _targeted_reseed(domain, triangulation, float(target_size),
                                  size_field, cancellation_check)
        targeted_reseed_count = len(reseed)
        if len(reseed):
            interior = np.vstack((interior, reseed))
            triangulation = triangulate_polygon(
                np.asarray(outer, dtype=float), holes=hole_polys,
                constraints=constraints, interior_points=interior,
                backend=_SEED_TRIANGULATION_BACKEND,
                cancellation_check=cancellation_check,
            )
    if len(triangulation.points) < len(outer) + sum(len(p) for p in hole_polys):
        raise MeshError("triangulation lost canonical boundary stations")
    for row, point in enumerate(outer):
        if not np.array_equal(triangulation.points[row], np.asarray(point, dtype=float)):
            raise MeshError("triangulation changed canonical outer station rows")
    cursor = len(outer)
    for hole_points in hole_polys:
        for offset, point in enumerate(hole_points):
            if not np.array_equal(triangulation.points[cursor + offset], np.asarray(point, dtype=float)):
                raise MeshError("triangulation changed canonical hole station rows")
        cursor += len(hole_points)
    nodes = {row: tuple(map(float, point)) for row, point in enumerate(triangulation.points)}
    cells = {row: tuple(map(int, tri)) for row, tri in enumerate(triangulation.triangles)}
    for body in cells.values():
        if orient2d(nodes[body[0]], nodes[body[1]], nodes[body[2]]) <= 0.0:
            raise MeshError("PQ2 seed contains a non-positive T3")
    boundary_edges = tuple(tuple(map(int, edge)) for edge in triangulation.boundary_segments)
    boundary_nodes = tuple(range(len(outer) + sum(len(p) for p in hole_polys)))
    point_tol = 1e-10 * max(1.0, float(np.ptp(triangulation.points, axis=0).max()))
    central_nodes = tuple(
        next(
            row for row, coordinate in enumerate(triangulation.points)
            if np.linalg.norm(coordinate - point) <= point_tol
        )
        for point in central_points
    )
    state = QuadMeshState(
        nodes, cells, {cid: "T3" for cid in cells},
        initial_front=boundary_edges,
        protected_nodes=boundary_nodes + central_nodes,
        protected_edges=boundary_edges + tuple(
            tuple(map(int, edge)) for edge in triangulation.mandatory_segments
        ),
    )
    station_to_node: dict[BoundaryStationKey, int] = {}
    for row, station in enumerate(stations):
        station_to_node[station.key] = row
    cursor = len(stations)
    for index in range(len(domain.hole_edge_uses)):
        hole_stations = registry.hole_stations(domain, index)
        for station in hole_stations:
            if station.key in station_to_node:
                raise MeshError("canonical boundary station identities are not unique on the face")
            station_to_node[station.key] = cursor
            cursor += 1
    domain.assert_current(geometry); registry.assert_current(geometry)
    if int(geometry.revision) != start_revision or str(geometry.model_id) != start_model_id:
        raise MeshError("PQ2 seed mutated source geometry")
    return PlanarQuadSeed(domain, registry, float(target_size), tuple(station_to_node),
                          triangulation, state, station_to_node, targeted_reseed_count)
