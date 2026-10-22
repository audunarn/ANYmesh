"""Station-conforming tensor Q4 seed across collinear split source edges.

The default quad-first layout used to seed every planar face with a size-driven
lattice and a constrained triangulation, even when the face is a plain
rectangle whose boundary stations already define a conforming grid.  Plates
whose sides are cut by stiffeners or junctions (several collinear source edges
per side) then showed fans and sheared patches because the lattice ignored the
station rows.

``edge_aligned_seed_data`` recognises a rigid rectangle whose sides may each be
made of several collinear straight source edges.  The four true corners are
found from the source edge directions; split vertices become ordinary stations
of a side.  Opposite sides must carry identical station fractions, then every
interior grid line passes exactly through one station on each opposite side and
the seed is a regular Q4 tensor with no interior irregular node.

Station rows and positions are the authority: this route never resamples,
moves, drops or adds boundary stations.  Anything it cannot represent exactly
(holes, angled or curved edges, nonuniform size fields, unequal opposite
counts or fractions, extreme cell aspect, oversized grids) declines with
``None`` and the caller keeps its established seed route untouched.
"""
from __future__ import annotations

import numpy as np
from anygeometry.curves import Straight
from anygeometry.model import GeometryModel

from ..refinement import SizeField
from ..triangulation import PlanarTriangulation
from .boundary import BoundaryStationRegistry
from .domain import PlanarQuadDomain
from .state import QuadMeshState

# Largest tensor node count the route will allocate; larger products decline.
_MAX_TENSOR_NODES = 250_000
# Source edges whose directions differ by less than this are one collinear side.
# Classification only: the exact station-on-side checks below are the gate.
_COLLINEAR_SINE = 1.0e-9
# Cells stretched beyond this ratio are left to the established seed route.
_MAX_CELL_ASPECT = 4.0


def _collinear(a: np.ndarray, b: np.ndarray) -> bool:
    na, nb = float(np.hypot(*a)), float(np.hypot(*b))
    if na <= 0.0 or nb <= 0.0:
        return False
    return (float(np.dot(a, b)) > 0.0
            and abs(float(a[0]*b[1] - a[1]*b[0])) <= _COLLINEAR_SINE * na * nb)


def edge_aligned_seed_data(geometry, domain, registry, outer, size_field,
                           cancellation_check, *, target_size):
    """Return ``(coverage, resident Q4 state)`` or ``None`` to decline.

    Inputs are never modified.  ``outer`` is the canonical station ring in
    chart coordinates, row-for-row with ``registry.outer_stations(domain)``.
    """
    if (type(geometry) is not GeometryModel
            or type(domain) is not PlanarQuadDomain
            or type(registry) is not BoundaryStationRegistry
            or domain.hole_edge_uses or len(domain.edge_uses) < 4
            or (size_field is not None
                and (type(size_field) is not SizeField or not size_field.is_uniform))
            or any(type(geometry.edges[e].curve) is not Straight
                   for e, _ in domain.edge_uses)):
        return None
    chains = [registry.chain(edge, forward) for edge, forward in domain.edge_uses]
    counts = [len(chain) - 1 for chain in chains]
    if any(count < 1 for count in counts):
        return None
    ring = np.asarray(outer, dtype=float)
    n = int(sum(counts))
    if ring.ndim != 2 or ring.shape != (n, 2) or not np.isfinite(ring).all():
        return None
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    # Direction of every source edge from its own end stations.  Interior
    # stations of a straight edge never decide where the corners are.
    ends = (starts + np.asarray(counts, dtype=np.int64)) % n
    directions = ring[ends] - ring[starts]
    corner_edges = [k for k in range(len(chains))
                    if not _collinear(directions[k - 1], directions[k])]
    if len(corner_edges) != 4:
        return None
    # ``starts`` ascends with the edge index, so the corners are in ring order
    # even when the loop begins in the middle of a side.
    c0, c1, c2, c3 = (int(starts[k]) for k in corner_edges)
    nb, nr, nt, nl = c1 - c0, c2 - c1, c3 - c2, c0 + n - c3
    if nb != nt or nr != nl:
        return None
    nx, ny = nb, nr
    if nx < 1 or ny < 1 or (nx + 1) * (ny + 1) > _MAX_TENSOR_NODES:
        return None
    corners = ring[[c0, c1, c2, c3]]
    u, v = corners[1] - corners[0], corners[3] - corners[0]
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    epsilon = 64 * np.finfo(float).eps
    tolerance = epsilon * max(1., lu, lv)
    if (lu <= tolerance or lv <= tolerance
            or abs(float(np.dot(u, v))) > epsilon * lu * lv
            or np.linalg.norm(corners[2] - (corners[0] + u + v)) > tolerance
            or float(u[0]*v[1] - u[1]*v[0]) <= 0.):
        return None
    bottom = ring[(c0 + np.arange(nb + 1)) % n]
    right = ring[(c1 + np.arange(nr + 1)) % n]
    top = ring[(c2 + np.arange(nt + 1)) % n][::-1]
    left = ring[(c3 + np.arange(nl + 1)) % n][::-1]
    # Every station on a side lies on that side and opposite sides carry the
    # same fractions: count equality alone would silently lose station pins.
    alpha = (bottom - corners[0]) @ u / (lu * lu)
    beta = (left - corners[0]) @ v / (lv * lv)
    if (not np.all(np.diff(alpha) > 0.) or not np.all(np.diff(beta) > 0.)
            or np.max(np.linalg.norm(bottom - (corners[0] + alpha[:, None]*u), axis=1)) > tolerance
            or np.max(np.linalg.norm(top - (corners[3] + alpha[:, None]*u), axis=1)) > tolerance
            or np.max(np.linalg.norm(left - (corners[0] + beta[:, None]*v), axis=1)) > tolerance
            or np.max(np.linalg.norm(right - (corners[1] + beta[:, None]*v), axis=1)) > tolerance):
        return None
    widths, heights = np.diff(alpha) * lu, np.diff(beta) * lv
    if (float(np.max(widths)) > _MAX_CELL_ASPECT * float(np.min(heights))
            or float(np.max(heights)) > _MAX_CELL_ASPECT * float(np.min(widths))):
        return None
    if cancellation_check is not None:
        cancellation_check("quad-first:edge-aligned-tensor")
    points = [tuple(map(float, point)) for point in outer]
    grid = np.empty((ny + 1, nx + 1), dtype=np.int64)
    grid[0, :] = (c0 + np.arange(nx + 1)) % n
    grid[:, nx] = (c1 + np.arange(ny + 1)) % n
    grid[ny, :] = (c2 + np.arange(nx, -1, -1)) % n
    grid[:, 0] = (c3 + np.arange(ny, -1, -1)) % n
    for j in range(1, ny):
        if cancellation_check is not None:
            cancellation_check("quad-first:edge-aligned-tensor-row")
        row = corners[0] + alpha[1:nx, None]*u + beta[j]*v
        grid[j, 1:nx] = len(points) + np.arange(nx - 1)
        points.extend(tuple(map(float, point)) for point in row)
    coordinates = np.asarray(points, dtype=float)
    a, b = grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel()
    c, d = grid[1:, 1:].ravel(), grid[1:, :-1].ravel()
    bodies = np.stack((a, b, c, d), axis=1)
    for k in range(4):
        p = coordinates[bodies[:, k]]
        q = coordinates[bodies[:, (k + 1) % 4]]
        r = coordinates[bodies[:, (k + 2) % 4]]
        if np.any((q[:, 0]-p[:, 0])*(r[:, 1]-p[:, 1])
                  - (q[:, 1]-p[:, 1])*(r[:, 0]-p[:, 0]) <= 0.):
            return None
    cells = {}
    for cid, body in enumerate(bodies.tolist()):
        if cancellation_check is not None and cid % 256 == 0:
            cancellation_check("quad-first:edge-aligned-tensor-cells")
        cells[cid] = tuple(int(node) for node in body)
    triangles = np.empty((2 * len(bodies), 3), dtype=np.int64)
    triangles[0::2] = np.stack((a, b, c), axis=1)
    triangles[1::2] = np.stack((a, c, d), axis=1)
    boundary = np.asarray([(i, (i + 1) % n) for i in range(n)], dtype=np.int64)
    coverage = PlanarTriangulation(
        points=coordinates, triangles=triangles,
        segments=boundary, boundary_segments=boundary,
        mandatory_segments=np.empty((0, 2), dtype=np.int64),
        outer_loop=np.arange(n, dtype=np.int64), hole_loops=(),
        backend="structured-rectangle", requested_backend="structured-rectangle",
        selected_backend="structured-rectangle", actual_backend="structured-rectangle",
    )
    state = QuadMeshState(dict(enumerate(points)), cells, {cid: "Q4" for cid in cells},
                          initial_front=(), protected_nodes=range(n),
                          protected_edges=tuple(tuple(map(int, edge)) for edge in boundary))
    return coverage, state
