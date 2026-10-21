"""Hidden planar tensor-background/CDT prototype; not a public meshing strategy.

Authored polygon stations are authoritative. Regular cells separated from all
authored segments survive unchanged; a GLOBAL constrained triangulation repairs
the remainder. This deliberately favors an auditable first slice over claiming
local-cavity performance. No geometry wheel or tolerance changes are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import ceil, hypot, isfinite
from typing import Any

import numpy as np

from .errors import MeshError
from .triangulation import orient2d, triangulate_polygon, _point_on_segment


@dataclass(frozen=True)
class SmartMappedPrototype:
    points: np.ndarray
    quads: np.ndarray
    triangles: np.ndarray
    outer_loop: np.ndarray
    hole_loops: tuple[np.ndarray, ...]
    constraints: np.ndarray
    constraint_chains: tuple[np.ndarray, ...]
    diagnostics: dict[str, Any]


class _Work:
    def __init__(self, limit, cancellation_check):
        self.count = 0
        self.limit = limit
        self.check = cancellation_check

    def step(self, stage="smart-mapped:classification"):
        self.count += 1
        if self.count > self.limit:
            raise MeshError("smart mapped prototype exceeds geometric work budget")
        if self.check is not None and self.count % 128 == 0:
            self.check(stage)


def _on(point, a, b):
    return (min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= point[1] <= max(a[1], b[1])
            and orient2d(a, b, point) == 0.)


def _intersects(a, b, c, d):
    ab_c, ab_d = orient2d(a,b,c), orient2d(a,b,d)
    cd_a, cd_b = orient2d(c,d,a), orient2d(c,d,b)
    return ((ab_c > 0. > ab_d or ab_d > 0. > ab_c)
            and (cd_a > 0. > cd_b or cd_b > 0. > cd_a)
            or (ab_c == 0. and _on(c,a,b)) or (ab_d == 0. and _on(d,a,b))
            or (cd_a == 0. and _on(a,c,d)) or (cd_b == 0. and _on(b,c,d)))


def _touch_only_at_shared_endpoint(a,b,c,d):
    common = set((a,b)) & set((c,d))
    return len(common) == 1 and len({p for p in (a,b,c,d) if _on(p,a,b) and _on(p,c,d)}) == 1


def _edges(loop):
    return tuple(zip(loop, loop[1:]+loop[:1]))


def _location(point, loop, work):
    winding = 0
    for a,b in _edges(loop):
        work.step()
        if _on(point,a,b):
            return 0
        if a[1] <= point[1] < b[1] and orient2d(a,b,point) > 0.:
            winding += 1
        elif b[1] <= point[1] < a[1] and orient2d(a,b,point) < 0.:
            winding -= 1
    return 1 if winding else -1


def _ring(raw, name, remaining):
    if isinstance(raw,np.ndarray) and (raw.ndim != 2 or raw.shape[1:] != (2,)):
        raise MeshError(f"{name} must contain finite planar coordinates")
    if len(raw) > remaining+1:
        raise MeshError("smart mapped prototype exceeds boundary station budget")
    if any(len(row) != 2 for row in raw):
        raise MeshError(f"{name} must contain finite planar coordinates")
    values = np.array(raw,dtype=np.float64,copy=True)
    if values.ndim != 2 or values.shape[1:] != (2,) or not np.all(np.isfinite(values)):
        raise MeshError(f"{name} must contain finite planar coordinates")
    loop = tuple(tuple(map(float,row)) for row in values)
    if len(loop) > 1 and loop[-1] == loop[0]:
        loop = loop[:-1]
    if len(loop) < 3 or len(set(loop)) != len(loop):
        raise MeshError(f"{name} needs at least three distinct stations")
    return loop


def _area2(loop):
    points = tuple((Fraction(x),Fraction(y)) for x,y in loop)
    return sum((a[0]*b[1]-a[1]*b[0] for a,b in _edges(points)),Fraction())


def _segment_hits_box(a,b,box):
    x0,y0 = box[0]
    x1,y1 = box[2]
    if (max(a[0],b[0]) < x0 or min(a[0],b[0]) > x1
            or max(a[1],b[1]) < y0 or min(a[1],b[1]) > y1):
        return False
    if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (a,b)):
        return True
    return any(_intersects(a,b,c,d) for c,d in _edges(box))


def build_smart_mapped(outer, holes=(), constraints=(), *, target_size,
                       max_grid_cells=4096, max_work=20_000_000,
                       cancellation_check=None, enable_pairing=True,
                       enable_smoothing=True, improvement_budget=8192,
                       enable_refinement=True, refinement_budget=16):
    """Return fixed-station Q4/T3 material for a simple planar polygon.

    Constraints are straight endpoint pairs, sampled to the nominal target size.
    Ordered ``constraint_chains`` preserve source order and endpoints; floating
    interpolation can differ from the ideal straight line by rounding. Outer and
    hole stations stay unchanged. Crossings, overlaps, touching holes
    and an internal endpoint meeting an unsampled boundary station are refused.
    The existing CDT's tolerance/refusal policy remains authoritative; exact
    post-audits additionally refuse any station/segment alteration or area gap.
    Array outputs are detached and read-only. No public registration is implied.
    """
    if cancellation_check is not None:
        cancellation_check("smart-mapped:start")
    if (type(max_grid_cells) is not int or max_grid_cells < 1
            or type(max_work) is not int or max_work < 1):
        raise MeshError("smart mapped prototype budgets must be positive integers")
    if (type(enable_pairing) is not bool or type(enable_smoothing) is not bool
            or type(improvement_budget) is not int or not 0 <= improvement_budget <= 100_000):
        raise MeshError("invalid prototype improvement controls or bounded budget")
    if (type(enable_refinement) is not bool or type(refinement_budget) is not int
            or not 0 <= refinement_budget <= 64):
        raise MeshError("invalid prototype refinement controls or bounded budget")
    h = float(target_size)
    if not isfinite(h) or h <= 0.:
        raise MeshError("smart mapped prototype target_size must be positive and finite")
    work = _Work(max_work,cancellation_check)
    outer = _ring(outer,"outer",4096)
    made_holes = []
    boundary_count = len(outer)
    for hole in holes:
        made = _ring(hole,"hole",4096-boundary_count)
        boundary_count += len(made)
        if boundary_count > 4096:
            raise MeshError("smart mapped prototype exceeds boundary station budget")
        made_holes.append(made)
    holes = tuple(made_holes)
    loops = (outer,*holes)
    if sum(map(len,loops)) > 4096:
        raise MeshError("smart mapped prototype exceeds boundary station budget")
    for loop in loops:
        if not _area2(loop):
            raise MeshError("smart mapped prototype refuses zero-area loops")
        edges = _edges(loop)
        for i,(a,b) in enumerate(edges):
            for j in range(i+1,len(edges)):
                work.step("smart-mapped:input-validation")
                c,d = edges[j]
                if _intersects(a,b,c,d) and not _touch_only_at_shared_endpoint(a,b,c,d):
                    raise MeshError("smart mapped prototype requires simple nonoverlapping loop edges")
    for i,loop in enumerate(loops):
        for other in loops[i+1:]:
            for a,b in _edges(loop):
                for c,d in _edges(other):
                    work.step("smart-mapped:input-validation")
                    if _intersects(a,b,c,d):
                        raise MeshError("smart mapped prototype refuses touching/intersecting loops")
    for i,hole in enumerate(holes):
        if _location(hole[0],outer,work) != 1:
            raise MeshError("hole is not strictly inside the outer loop")
        if any(_location(hole[0],other,work) >= 0 or _location(other[0],hole,work) >= 0
               for other in holes[i+1:]):
            raise MeshError("smart mapped prototype refuses nested/overlapping holes")
    authored_edges = tuple(edge for loop in loops for edge in _edges(loop))
    boundary_points = set(point for loop in loops for point in loop)

    def material(point):
        return (_location(point,outer,work) == 1
                and all(_location(point,hole,work) == -1 for hole in holes))

    segments = []
    for raw in constraints:
        if isinstance(raw,np.ndarray) and raw.shape != (2,2):
            raise MeshError("each internal constraint must have two finite planar endpoints")
        if len(raw) != 2 or any(len(row) != 2 for row in raw):
            raise MeshError("each internal constraint must have two finite planar endpoints")
        values = np.array(raw,dtype=np.float64,copy=True)
        if values.shape != (2,2) or not np.all(np.isfinite(values)):
            raise MeshError("each internal constraint must have two finite planar endpoints")
        a,b = (tuple(map(float,row)) for row in values)
        if a == b:
            raise MeshError("zero-length internal constraint")
        if any(point not in boundary_points and not material(point) for point in (a,b)):
            raise MeshError("constraint endpoint must be material or an existing boundary station")
        for c,d in (*authored_edges,*segments):
            work.step("smart-mapped:input-validation")
            if _intersects(a,b,c,d) and not _touch_only_at_shared_endpoint(a,b,c,d):
                raise MeshError("crossing/overlapping constraints are unsupported by this prototype")
        segments.append((a,b))
        if len(segments) > 4096:
            raise MeshError("smart mapped prototype exceeds constraint budget")
    # Preflight the complete derived station count before allocating any chain.
    # The fixed station cap also bounds a long line at a very small target size.
    counts = []
    station_count = 0
    for a,b in segments:
        work.step("smart-mapped:constraint-refinement")
        ratio = hypot(b[0]-a[0],b[1]-a[1])/h
        if not isfinite(ratio) or ratio > 4096:
            raise MeshError("smart mapped prototype exceeds refined constraint station budget")
        count = max(1,ceil(ratio))
        station_count += count+1
        if station_count > 4096:
            raise MeshError("smart mapped prototype exceeds refined constraint station budget")
        counts.append(count)
    chains = []
    for (a,b),count in zip(segments,counts):
        chain = [a]
        for i in range(1,count):
            work.step("smart-mapped:constraint-refinement")
            t = i/count
            chain.append((a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t))
        chain.append(b)
        if len(set(chain)) != len(chain):
            raise MeshError("constraint refinement cannot resolve distinct stations")
        chains.append(tuple(chain))
    segments = tuple(edge for chain in chains for edge in zip(chain,chain[1:]))
    x0,x1 = min(p[0] for p in outer),max(p[0] for p in outer)
    y0,y1 = min(p[1] for p in outer),max(p[1] for p in outer)
    if not all(isfinite(value) and value > 0. for value in ((x1-x0)/h,(y1-y0)/h)):
        raise MeshError("degenerate or unbounded tensor background")
    nx,ny = ceil((x1-x0)/h),ceil((y1-y0)/h)
    if nx*ny > max_grid_cells:
        raise MeshError("smart mapped prototype exceeds background cell budget")
    xs,ys = np.linspace(x0,x1,nx+1),np.linspace(y0,y1,ny+1)
    # Same default membership envelope as triangulation._prepare_pslg. It is
    # used only to decline optional grid candidates, never to move source data.
    # Binary samples of a straight side need not be exactly collinear.
    cdt_epsilon = max(x1-x0,y1-y0,1.)*1.0e-12
    grid = tuple((float(x),float(y)) for y in ys for x in xs)
    retained = []
    retained_indices = []
    barriers = (*authored_edges,*segments)
    for j in range(ny):
        if cancellation_check is not None:
            cancellation_check("smart-mapped:background-row")
        for i in range(nx):
            first = j*(nx+1)+i
            box = tuple(grid[row] for row in (first,first+1,first+nx+2,first+nx+1))
            # Full-segment intersection, not cell-center deletion: a tiny hole
            # wholly inside one background cell still removes that cell.
            hit = False
            for a,b in barriers:
                work.step()
                if (_segment_hits_box(a,b,box)
                        or any(_point_on_segment(point,a,b,cdt_epsilon) for point in box)):
                    hit = True
                    break
            if not hit and material(box[0]):
                retained.append(box)
                retained_indices.append((i,j))
    retained_edges = tuple(sorted({tuple(sorted(edge)) for box in retained for edge in _edges(box)}))
    extras = dict.fromkeys(point for segment in segments for point in segment if point not in boundary_points)
    required_grid_points = {point for box in retained for point in box}
    for point in grid:
        work.step("smart-mapped:interior-stations")
        # Unrequired free grid points in the transition region can create tiny
        # edges beside prescribed stations. Leave the actual CDT to span this
        # region; retain every distant quad and every authored endpoint.
        if point not in required_grid_points or point in boundary_points or point in extras:
            continue
        extras[point] = None
    if cancellation_check is not None:
        cancellation_check("smart-mapped:global-cdt")
    # Match triangulate_polygon's exact input block order: loops, extras, then
    # every constraint endpoint occurrence. Duplicate coordinates share their
    # canonical ID; distinct nearby stations must never silently collapse.
    all_constraints = (*segments,*retained_edges)
    input_points = (tuple(point for loop in loops for point in loop)+tuple(extras)
                    +tuple(point for segment in all_constraints for point in segment))
    canonical = {}
    protected = {}
    for row,point in enumerate(input_points):
        protected[row] = canonical.setdefault(point,len(canonical))
    cdt = triangulate_polygon(np.asarray(outer),tuple(np.asarray(hole) for hole in holes),
        tuple(np.asarray(edge) for edge in all_constraints),
        interior_points=np.asarray(tuple(extras),dtype=float).reshape((-1,2)),
        backend="python",cancellation_check=cancellation_check,
        protected_node_ids=protected)
    point_ids = {tuple(map(float,point)):row for row,point in enumerate(cdt.points)}
    if len(point_ids) != len(cdt.points):
        raise MeshError("CDT returned duplicate coordinates")
    try:
        outer_ids = tuple(point_ids[p] for p in outer)
        hole_ids = tuple(tuple(point_ids[p] for p in hole) for hole in holes)
        constraint_ids = tuple((point_ids[a],point_ids[b]) for a,b in segments)
        chain_ids = tuple(tuple(point_ids[p] for p in chain) for chain in chains)
        quads = tuple(tuple(point_ids[p] for p in box) for box in retained)
    except KeyError as error:
        raise MeshError("CDT changed a prescribed station") from error
    covered = {}
    for row,quad in enumerate(quads):
        for triple in combinations(quad,3):
            covered[frozenset(triple)] = row
    removed = [[] for _ in quads]
    triangles = []
    for raw in cdt.triangles:
        work.step("smart-mapped:quad-recovery")
        triangle = tuple(map(int,raw))
        owner = covered.get(frozenset(triangle))
        if owner is None:
            triangles.append(triangle)
        else:
            removed[owner].append(triangle)
    for quad,parts in zip(quads,removed):
        diagonal = set(parts[0]) & set(parts[1]) if len(parts) == 2 else set()
        covered_area = sum((_area2(tuple(tuple(cdt.points[n]) for n in tri))
                            for tri in parts),Fraction())
        if (len(parts) != 2 or len(diagonal) != 2
                or frozenset(diagonal) in {frozenset(edge) for edge in _edges(quad)}
                or covered_area != _area2(tuple(tuple(cdt.points[n]) for n in quad))):
            raise MeshError("CDT did not exactly cover a retained tensor cell")
    from ._smart_mapped_improve import improve
    fixed_nodes = {node for loop in (outer_ids,*hole_ids,*chain_ids) for node in loop}
    protected_edges = {tuple(sorted(edge)) for loop in (outer_ids,*hole_ids) for edge in _edges(loop)}
    protected_edges.update(tuple(sorted(edge)) for edge in constraint_ids)
    result_points,quads,triangles,improvement = improve(cdt.points,quads,triangles,
        fixed_nodes=fixed_nodes,protected_edges=protected_edges,
        enable_pairing=enable_pairing,enable_smoothing=enable_smoothing,
        budget=improvement_budget,work=work,cancellation_check=cancellation_check)
    from ._smart_mapped_refine import refine
    result_points,quads,triangles,refinement = refine(result_points,quads,triangles,
        protected_edges=protected_edges,enabled=enable_refinement,budget=refinement_budget,
        work=work,cancellation_check=cancellation_check)
    cells = (*quads,*triangles)
    incidence = {}
    total = Fraction()
    for body in cells:
        work.step("smart-mapped:coverage-audit")
        points = tuple(tuple(map(float,result_points[n])) for n in body)
        area = _area2(points)
        if area <= 0 or any(orient2d(points[i-1],points[i],points[(i+1)%len(points)]) <= 0
                            for i in range(len(points))):
            raise MeshError("nonpositive cell in smart mapped prototype")
        total += area
        for a,b in _edges(body):
            edge = tuple(sorted((a,b)))
            incidence[edge] = incidence.get(edge,0)+1
    expected_boundary = {tuple(sorted(edge)) for loop in (outer_ids,*hole_ids) for edge in _edges(loop)}
    if ({edge for edge,count in incidence.items() if count == 1} != expected_boundary
            or any(count not in (1,2) for count in incidence.values())):
        raise MeshError("mixed mesh changed prescribed boundaries or has hanging edges")
    if any(incidence.get(tuple(sorted(edge))) != 2 for edge in constraint_ids):
        raise MeshError("mixed mesh did not preserve an internal constraint segment")
    expected_area = abs(_area2(outer))-sum((abs(_area2(hole)) for hole in holes),Fraction())
    if total != expected_area:
        raise MeshError("mixed mesh exact coverage area differs from authored material")
    used = sorted({node for body in cells for node in body})
    remap = {old:new for new,old in enumerate(used)}
    def array(values,width=None):
        result = np.asarray(values,dtype=np.int64)
        if width is not None:
            result = result.reshape((-1,width))
        result.setflags(write=False)
        return result
    improvement['moved_node_ids'] = [remap[n] for n in improvement['moved_node_ids']]
    points = np.array(result_points[used],copy=True)
    points.setflags(write=False)
    if cancellation_check is not None:
        cancellation_check("smart-mapped:complete")
    return SmartMappedPrototype(points,
        array([[remap[n] for n in body] for body in quads],4),
        array([[remap[n] for n in body] for body in triangles],3),
        array([remap[n] for n in outer_ids]),
        tuple(array([remap[n] for n in loop]) for loop in hole_ids),
        array([[remap[n] for n in edge] for edge in constraint_ids],2),
        tuple(array([remap[n] for n in chain]) for chain in chain_ids),
        dict(prototype=True,repair_scope="global_constrained_triangulation",
             quality_status="research_candidate_not_production_adjudicated",
             background_cells=nx*ny,retained_quads=len(retained),triangles=len(triangles),
             improvement=improvement,refinement=refinement,
             retained_grid_indices=retained_indices,grid_shape=(ny,nx),
             source_constraints=len(chains),derived_constraint_segments=len(segments),
             exact_coverage=True,prescribed_boundaries_exact=True,geometric_work=work.count,
             actual_backend=cdt.actual_backend))
