"""Station-conforming tensor cells for the narrow planar rectangular case."""
from __future__ import annotations

import numpy as np
from anygeometry.curves import Straight
from anygeometry.model import GeometryModel

from ..refinement import SizeField
from ..errors import MeshError
from ..triangulation import PlanarTriangulation, orient2d
from .boundary import BoundaryStationRegistry
from .domain import PlanarQuadDomain
from .state import QuadMeshState


def rectangle_seed_data(geometry, domain, registry, outer, size_field,
                        cancellation_check, *, target_size):
    """Return coverage and resident Q4 cells, or decline without changing inputs.

    Existing stations are the authority. This route never balances, resamples,
    moves or removes them. A roundoff-scale admission check allows rigidly
    transformed rectangles; it is not a replacement for downstream quality,
    area, topology or high-order validity checks.
    """
    if (type(geometry) is not GeometryModel
            or type(domain) is not PlanarQuadDomain
            or type(registry) is not BoundaryStationRegistry
            or domain.hole_edge_uses or len(domain.edge_uses) != 4
            or (size_field is not None
                and (type(size_field) is not SizeField or not size_field.is_uniform))
            or any(type(geometry.edges[e].curve) is not Straight
                   for e, _ in domain.edge_uses)):
        return None
    chains = [registry.chain(edge, forward) for edge, forward in domain.edge_uses]
    if len(chains[0]) != len(chains[2]) or len(chains[1]) != len(chains[3]):
        return None
    sides = [np.asarray([domain.project(s.position) for s in chain], dtype=float)
             for chain in chains]
    if any(not np.isfinite(side).all() for side in sides):
        return None
    corners = np.asarray([side[0] for side in sides])
    u, v = corners[1] - corners[0], corners[3] - corners[0]
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    epsilon = 64 * np.finfo(float).eps
    tolerance = epsilon * max(1., lu, lv)
    if (not np.isfinite(corners).all() or lu <= tolerance or lv <= tolerance
            or abs(float(np.dot(u, v))) > epsilon * lu * lv
            or np.linalg.norm(corners[2] - (corners[0] + u + v)) > tolerance
            or orient2d(corners[0], corners[1], corners[3]) <= 0.):
        return None
    # Align opposite sides to the same tensor direction. Both their counts
    # and every prescribed station fraction must match; count equality alone
    # would silently lose nonuniform station controls.
    bottom, right, top, left = sides[0], sides[1], sides[2][::-1], sides[3][::-1]
    alpha = (bottom - corners[0]) @ u / (lu * lu)
    beta = (left - corners[0]) @ v / (lv * lv)
    if (not np.all(np.diff(alpha) > 0.) or not np.all(np.diff(beta) > 0.)
            or np.max(np.linalg.norm(bottom - (corners[0] + alpha[:, None]*u), axis=1)) > tolerance
            or np.max(np.linalg.norm(top - (corners[3] + alpha[:, None]*u), axis=1)) > tolerance
            or np.max(np.linalg.norm(left - (corners[0] + beta[:, None]*v), axis=1)) > tolerance
            or np.max(np.linalg.norm(right - (corners[1] + beta[:, None]*v), axis=1)) > tolerance):
        return None
    nx, ny = len(bottom)-1, len(left)-1
    # Preserve the adaptive route's fine-candidate refusal before allocating
    # a tensor grid. Use the requested size: registry compatibility permits
    # tiny differences, but ceil at the budget boundary is bit-sensitive.
    extent = np.max(np.asarray(outer), axis=0) - np.min(np.asarray(outer), axis=0)
    fine_step = float(target_size) / 2.0
    if not np.isfinite(fine_step) or fine_step <= 0.0:
        raise MeshError("adaptive seed size must be positive and finite")
    fine_counts = np.maximum(1, np.ceil(extent / fine_step).astype(int))
    if int(fine_counts[0]) * int(fine_counts[1]) > 250000:
        raise MeshError("adaptive seed exceeds fine candidate budget")
    # Dense explicit station pins can demand a much larger Cartesian product
    # than the target size. Decline to the original unstructured route rather
    # than allocating that product or rejecting previously supported pins.
    if (nx+1)*(ny+1) > 250000:
        return None
    if cancellation_check is not None:
        cancellation_check("quad-first:structured-rectangle")
    points = [tuple(map(float, point)) for point in outer]
    grid = np.empty((ny+1, nx+1), dtype=np.int64)
    offsets = np.cumsum([0, nx, ny, nx])
    grid[0, :] = np.arange(nx+1)
    grid[:, nx] = offsets[1] + np.arange(ny+1)
    grid[ny, :] = offsets[2] + np.arange(nx, -1, -1)
    grid[:, 0] = (offsets[3] + np.arange(ny, -1, -1)) % len(outer)
    for j in range(1, ny):
        if cancellation_check is not None:
            cancellation_check("quad-first:structured-rectangle-row")
        for i in range(1, nx):
            grid[j, i] = len(points)
            points.append(tuple(map(float, corners[0] + alpha[i]*u + beta[j]*v)))
    cells = {}
    triangles = []
    for j in range(ny):
        for i in range(nx):
            if cancellation_check is not None and i % 256 == 0:
                cancellation_check("quad-first:structured-rectangle-cells")
            body = tuple(map(int, (grid[j, i], grid[j, i+1], grid[j+1, i+1], grid[j+1, i])))
            if any(orient2d(points[body[k]], points[body[(k+1)%4]], points[body[(k+2)%4]]) <= 0.
                   for k in range(4)):
                return None
            cells[len(cells)] = body
            triangles.extend(((body[0], body[1], body[2]), (body[0], body[2], body[3])))
    boundary = np.asarray([(i, (i+1) % len(outer)) for i in range(len(outer))], dtype=np.int64)
    coverage = PlanarTriangulation(
        points=np.asarray(points), triangles=np.asarray(triangles, dtype=np.int64),
        segments=boundary, boundary_segments=boundary,
        mandatory_segments=np.empty((0, 2), dtype=np.int64),
        outer_loop=np.arange(len(outer), dtype=np.int64), hole_loops=(),
        backend="structured-rectangle", requested_backend="structured-rectangle",
        selected_backend="structured-rectangle", actual_backend="structured-rectangle",
    )
    state = QuadMeshState(dict(enumerate(points)), cells, {cid: "Q4" for cid in cells},
                          initial_front=(), protected_nodes=range(len(outer)),
                          protected_edges=tuple(tuple(map(int, edge)) for edge in boundary))
    return coverage, state
