"""Remove complete parent cells bounded by one source-authored NURBS inner loop."""
from collections import defaultdict
from fractions import Fraction
from itertools import combinations

import numpy as np

from .errors import MeshError
from ._stage_cancellation import check_stage
from .regional_source_faces import _digest, _face_signature, _native_definitions, _ordinary_face, _vertex_faces


def _sides(cell):
    return zip(cell, (*cell[1:], cell[0]))


def _edge(a, b):
    return (min(a, b), max(a, b))


def _area(points):
    return sum(a[0]*b[1]-b[0]*a[1] for a,b in _sides(points))


def _turn(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _convex(points):
    return all(_turn(points[i-1], p, points[(i+1)%len(points)]) > 0 for i,p in enumerate(points))


def _perimeter(cells):
    edges = defaultdict(list)
    for cell in cells:
        for a,b in _sides(cell):
            edges[_edge(a,b)].append((a,b))
    outside = {}
    for owners in edges.values():
        if len(owners) == 2 and owners[0] == tuple(reversed(owners[1])):
            continue
        if len(owners) != 1 or owners[0][0] in outside:
            return None
        a,b = owners[0]
        outside[a] = b
    if not outside:
        return None
    ring = [min(outside)]
    for _ in range(len(outside)-1):
        nxt = outside.get(ring[-1])
        if nxt is None or nxt in ring:
            return None
        ring.append(nxt)
    return ring if outside.get(ring[-1]) == ring[0] else None


def _neighbor_blocks(cells, index):
    """Disjoint four-cell vertex stars outside each side of the original Q4."""
    edge_cells, stars = defaultdict(list), defaultdict(list)
    for i,c in enumerate(cells):
        for a,b in _sides(c):
            edge_cells[_edge(a,b)].append(i)
            stars[a].append(i)
    blocks, seen = [], {index}
    for a,b in _sides(cells[index]):
        across = [i for i in edge_cells[_edge(a,b)] if i != index]
        if len(across) != 1 or len(cells[across[0]]) != 4:
            return None
        other = cells[across[0]]
        j = other.index(b)
        if other[(j+1)%4] != a:
            return None
        pivot = other[j-1]
        star = stars[pivot]
        if len(star) != 4 or seen.intersection(star) or any(len(cells[i]) != 4 for i in star):
            return None
        neighbors = {v for i in star for v in (cells[i][cells[i].index(pivot)-1],
                     cells[i][(cells[i].index(pivot)+1)%4])}
        if len(neighbors) != 4 or any(len(edge_cells[_edge(v,pivot)]) != 2 for v in neighbors):
            return None
        blocks.append((pivot,star))
        seen.update(star)
    return blocks


def _grade_blocks(rows, cells, blocks, support):
    created = []
    def mean_row(ids):
        uv = np.mean([rows[n][0] for n in ids], axis=0)
        rows.append([uv.tolist(), np.asarray(support.evaluate(*uv)).tolist()])
        return len(rows)-1
    for pivot, star in blocks:
        mids = {}
        for i in star:
            c = cells[i]
            j = c.index(pivot)
            a,b,d = c[j-1], c[(j+1)%4], c[(j+2)%4]
            for n in (a,b):
                if n not in mids:
                    mids[n] = mean_row((pivot,n))
            m = mean_row(c)
            created.extend(([mids[a],pivot,mids[b],m], [mids[b],b,d,m], [d,a,mids[a],m]))
    return created


def _annulus_quality(rows, cells, settings):
    from .core import MeshCore
    from .quality_v2 import evaluate_quality
    from .surface_mesh import _quality_threshold_report, _published_growth
    xyz = np.asarray([row[1] for row in rows])
    core = MeshCore(xyz, np.array([c for c in cells if len(c)==3],int).reshape(-1,3),
                    np.array([c for c in cells if len(c)==4],int).reshape(-1,4))
    return (_quality_threshold_report(evaluate_quality(core),settings)['accepted']
            and not _published_growth(core, settings.max_element_growth)[1])


def _exact_annulus_partition(original, new, points, boundary, hole):
    edges = defaultdict(list)
    for cell in new:
        for a,b in _sides(cell):
            edges[_edge(a,b)].append((a,b))
    previous = {_edge(a,b):(a,b) for a,b in _sides(boundary)}
    previous.update({_edge(a,b):(b,a) for a,b in _sides(hole)})
    exposed = {}
    for side, adjacent in edges.items():
        if len(adjacent)==2 and adjacent[0]==tuple(reversed(adjacent[1])):
            continue
        if len(adjacent)!=1:
            return False
        exposed[side] = adjacent[0]
    def area(row):
        return _area([points[n] for n in row])
    return (exposed==previous and sum(map(area,new))+area(hole)==sum(map(area,original)))


_REFINE_EVALUATIONS = 96


def _nominal(rows, cells, limit):
    """The app's sizing gate: some cell's longest physical side is within target + geometry tolerance."""
    return any(max(float(np.linalg.norm(np.subtract(rows[a][1], rows[b][1]))) for a,b in _sides(c)) <= limit
               for c in cells)


def _refine_ring(rows, exact_uv, cells, selected, proposed, support, hole, settings, limit, check):
    """Refine the annulus itself by midpoint stars of the once-proposed material.

    Every valence-four, interior, non-hole node of the proposed annular
    cells whose star holds only those ring cells and old cells is a candidate. The
    old cells of a chosen star join the replaced footprint (the expanded cut then
    has a new fixed outer boundary) and the pivot is reissued as a fresh row on the
    support. Disjoint stars are polled in a fixed order, smallest sets first, within
    a bounded number of evaluations. Only cells derived from ring cells (never
    purely exterior old-cell subdivisions) are accepted as sizing proof.
    Returns (replaced_rows, new_cells, rows, interface) or None.
    """
    kept = [i for i in range(len(cells)) if i not in selected]
    material = [cells[i] for i in kept] + proposed
    first = len(kept)
    ring = set(range(first, len(material)))
    stars, edge_cells = defaultdict(list), defaultdict(list)
    for i,c in enumerate(material):
        for a,b in _sides(c):
            edge_cells[_edge(a,b)].append(i)
            stars[a].append(i)
    candidates = []
    for pivot in sorted({n for i in ring for n in material[i]} - set(hole)):
        star = sorted(stars[pivot])
        if len(star) != 4 or not ring.intersection(star) or any(len(material[i]) != 4 for i in star):
            continue
        around = {v for i in star for v in (material[i][material[i].index(pivot)-1],
                  material[i][(material[i].index(pivot)+1)%4])}
        if len(around) == 4 and all(len(edge_cells[_edge(v,pivot)]) == 2 for v in around):
            candidates.append((pivot, star))
    tried = 0
    for size in range(1, len(candidates)+1):
        for subset in combinations(candidates, size):
            tried += 1
            if tried > _REFINE_EVALUATIONS:
                return None
            check_stage(check, 'source trim ring refinement')
            used = [i for _,star in subset for i in star]
            if len(set(used)) != len(used):
                continue
            fresh = list(rows)
            mapped = [list(c) for c in material]
            blocks = []
            for pivot, star in subset:
                uv = fresh[pivot][0]
                fresh.append([list(uv), np.asarray(support.evaluate(*uv)).tolist()])
                for i in star:
                    mapped[i] = [len(fresh)-1 if n == pivot else n for n in mapped[i]]
                blocks.append((len(fresh)-1, star))
            created = _grade_blocks(fresh, mapped, blocks, support)
            annular, position = [], 0
            for _, star in blocks:
                for i in star:
                    if i in ring:
                        annular.extend(created[3*position:3*position+3])
                    position += 1
            replaced = set(used)
            annular = [c for j,c in enumerate(proposed) if first+j not in replaced] + annular
            # Hole stations retain the authored rational pcurve evaluation.
            # Only the additional, freely chosen interior rows use stored UV.
            exact = exact_uv + [tuple(map(Fraction, row[0])) for row in fresh[len(rows):]]
            footprint = set(selected) | {kept[i] for i in replaced if i < first}
            interface = _perimeter([cells[i] for i in sorted(footprint)])
            if interface is None or any(not _convex([exact[n] for n in c]) for c in created):
                continue
            result = [proposed[j] for j in range(len(proposed)) if first+j not in replaced] + created
            if (_nominal(fresh, annular, limit)
                    and _exact_annulus_partition([cells[i] for i in sorted(footprint)], result, exact, interface, hole)
                    and _compact_quality(fresh, [cells[i] for i in range(len(cells)) if i not in footprint] + result,
                                         settings)):
                return sorted(footprint), result, fresh, interface
    return None


def _plan_annulus(source, data, base, by_uv, field, overrides, recombine, quality, check):
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, _IDENTITY
    from .seeding import edge_demand
    from .surface_mesh import SurfaceMeshOptions
    face = source.faces[data['face']]
    support = face.surface
    if not recombine or len(face.holes[0]) != 4:
        return None
    old = data['node_rows']
    rows = list(old)
    local = {}
    hole, rim = [], []
    for use in face.holes[0]:
        edge = source.edges[use.edge]
        curve = edge.curve
        if (type(curve) is not CurveOnSurface or curve.support != support or curve.transform != _IDENTITY
                or type(curve.pcurve) is not NurbsCurve or len(curve.pcurve.controls) != 2
                or curve.pcurve.degree != 1 or curve.pcurve.weights and len(set(curve.pcurve.weights)) != 1):
            return None
        if overrides.get(edge.id, max(1,round(edge_demand(source, edge.id, field)))) != 1:
            return None
        stations = []
        for t, vertex in ((0.,edge.start),(1.,edge.end)):
            p = tuple(curve.pcurve.exact_point(t))
            if p in by_uv or any(Fraction(float(x)) != x for x in p):
                return None
            xyz = [float(x) for x in curve.exact_point(t)]
            if not np.array_equal(xyz, source.vertex_position(vertex)):
                return None
            row = [[float(x) for x in p], xyz]
            if vertex not in local:
                local[vertex] = len(rows)
                rows.append(row)
            if rows[local[vertex]] != row:
                return None
            stations.append([t,local[vertex]])
        rim.append([edge.id,stations])
        hole.append(stations[0 if use.forward else 1][1])
    if len(local) != 4 or len(set(hole)) != 4:
        return None
    def points(indices):
        return [tuple(map(Fraction, rows[n][0])) for n in indices]
    if _area(points(hole)) < 0:
        hole.reverse()
    if not _convex(points(hole)):
        return None
    cells = data['cells']
    bounds = np.asarray([rows[n][0] for n in hole])
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    found = []
    for i,c in enumerate(cells):
        if i % 256 == 0:
            check_stage(check,'source trim annulus containing cell')
        if len(c) != 4:
            continue
        box = np.asarray([rows[n][0] for n in c])
        if np.any(box.min(axis=0) >= lower) or np.any(box.max(axis=0) <= upper):
            continue
        parent = points(c)
        if _convex(parent) and all(_turn(a,b,p)>0 for a,b in _sides(parent) for p in points(hole)):
            found.append(i)
    if len(found) != 1:
        return None
    index, = found
    parent = cells[index]
    annuli = []
    for k in range(4):
        inner = hole[k:] + hole[:k]
        ring = [[a,b,inner[(i+1)%4],inner[i]] for i,(a,b) in enumerate(_sides(parent))]
        if (all(_convex(points(c)) for c in ring)
                and sum(_area(points(c)) for c in ring) + _area(points(hole)) == _area(points(parent))):
            cost = sum(sum((a-b)**2 for a,b in zip(p,q)) for p,q in zip(points(parent), points(inner)))
            annuli.append((cost,ring))
    if not annuli:
        return None
    if quality is not None and not set(quality) <= {'max_element_growth','min_angle','max_angle',
                                                   'max_aspect_ratio','min_scaled_jacobian','max_warpage'}:
        return None
    settings = SurfaceMeshOptions(**(quality or {}))
    for grade in (False,True):
        blocks = _neighbor_blocks(cells,index) if grade else []
        if blocks is None:
            return None
        selected = {index} | {i for _, star in blocks for i in star}
        interface = _perimeter([cells[i] for i in sorted(selected)])
        if interface is None:
            continue
        cut_rows = list(rows)
        transitions = _grade_blocks(cut_rows, cells, blocks, support)
        exact = [tuple(map(Fraction,row[0])) for row in cut_rows]
        if any(not _convex([exact[n] for n in c]) for c in transitions):
            continue
        kept = [c for i,c in enumerate(cells) if i not in selected]
        for _,ring in sorted(annuli):
            proposed = ring+transitions
            check_stage(check,'source trim annulus physical quality')
            if (not _exact_annulus_partition([cells[i] for i in selected], proposed, exact, interface, hole)
                    or not _annulus_quality(cut_rows,kept+proposed,settings)):
                continue
            return dict(base, annulus=True, hole_cell_row=index, hole_cell_interface_rows=list(parent),
                        new_rows=cut_rows[len(old):], new_cells=proposed, parent_node_count=len(old),
                        replaced_cell_rows=sorted(selected), interface_rows=interface, hole_rim=rim,
                        quality=dict(quality or {}))
    return None


def _interiors_meet(p, q):
    """Exact test for overlap of two convex counter-clockwise polygon interiors."""
    return not (any(all(_turn(a,b,x) <= 0 for x in q) for a,b in _sides(p))
                or any(all(_turn(a,b,x) <= 0 for x in p) for a,b in _sides(q)))


def _derive_grid(face, data, check):
    """Derive node and cell rows from oriented Q4 incidence and the four ordered rim chains.

    Starting from the first rim chain, each step crosses the single not yet used cell
    on the far side of every edge of the current row. Nothing is inferred from
    coordinates. Returns None unless one complete disk, a node bijection, the
    consistently oriented interior and every outer edge are proved.
    """
    cells, count = data['cells'], len(data['node_rows'])
    try:
        stations = {edge: s for edge, s in data['rim']}
        chains = []
        for use in face.loop:
            ts, nodes = [t for t,_ in stations[use.edge]], [n for _,n in stations[use.edge]]
            if (len(ts) < 2 or ts[0] != 0 or ts[-1] != 1
                    or not all(a < b for a,b in zip(ts, ts[1:]))):
                return None
            chains.append(nodes if use.forward else nodes[::-1])
    except (KeyError, TypeError, ValueError):
        return None
    if (len(chains) != 4 or len(stations) != 4 or len(data['rim']) != 4
            or any(a[-1] != b[0] for a,b in zip(chains, (*chains[1:], chains[0])))):
        return None
    c0, c1, c2, c3 = chains
    A, B = len(c0)-1, len(c1)-1
    if len(c2) != A+1 or len(c3) != B+1 or len(cells) != A*B or any(len(c) != 4 for c in cells):
        return None
    incidence = defaultdict(list)
    for row, cell in enumerate(cells):
        if row % 256 == 0:
            check_stage(check, 'source trim grid derivation')
        if len(set(cell)) != 4:
            return None
        for a,b in _sides(cell):
            incidence[_edge(a,b)].append((row,a,b))
    outer = {_edge(a,b) for c in chains for a,b in zip(c, c[1:])}
    exposed = set()
    for edge, adjacent in incidence.items():
        if len(adjacent) == 1:
            exposed.add(edge)
        elif len(adjacent) != 2 or adjacent[0][1:] != adjacent[1][:0:-1]:
            return None
    if exposed != outer:
        return None
    def beyond(cell, node, other):
        j = cell.index(node)
        side = [cell[(j-1)%4], cell[(j+1)%4]]
        if other not in side:
            return None
        return side[0] if side[1] == other else side[1]
    levels, used, cell_at = [list(c0)], set(), {}
    for b in range(B):
        check_stage(check, 'source trim grid derivation')
        below, above = levels[-1], [None]*(A+1)
        for i in range(A):
            u, v = below[i], below[i+1]
            fresh = [r for r,_,_ in incidence.get(_edge(u,v), ()) if r not in used]
            if len(fresh) != 1:
                return None
            used.add(fresh[0])
            cell_at[i,b] = fresh[0]
            cell = cells[fresh[0]]
            pu, pv = beyond(cell, u, v), beyond(cell, v, u)
            if pu is None or pv is None or above[i] not in (None, pu) or above[i+1] is not None:
                return None
            above[i], above[i+1] = pu, pv
        if None in above:
            return None
        levels.append(above)
        if b+1 < B and (above[0] != c3[B-b-1] or above[-1] != c1[b+1]):
            return None
    if (levels[-1] != c2[::-1] or any(l[0] != c3[B-b] or l[-1] != c1[b] for b,l in enumerate(levels))
            or len(used) != len(cells)):
        return None
    flat = [n for level in levels for n in level]
    if len(flat) != count or set(flat) != set(range(count)):
        return None
    return np.asarray(levels, dtype=int).T, cell_at


def _hole_chains(source, face, support, field, overrides, by_uv, rows, exact_new, max_divisions):
    """Append the authored hole's actual owner stations to rows; return CCW corner-to-corner chains.

    exact_new is parallel to rows and receives the exact rational UV of every appended row;
    rows only hold the binary64 projection of it, so non-dyadic stations never leak into the proofs.
    """
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, _IDENTITY
    from .hybrid import _station_parameters
    from .seeding import edge_demand
    local, rim, chains, fresh = {}, [], [], set()
    if len({use.edge for use in face.holes[0]}) != 4:
        return None
    for use in face.holes[0]:
        edge = source.edges[use.edge]
        curve = edge.curve
        if (type(curve) is not CurveOnSurface or curve.support != support or curve.transform != _IDENTITY
                or type(curve.pcurve) is not NurbsCurve or len(curve.pcurve.controls) != 2
                or curve.pcurve.degree != 1 or curve.pcurve.weights and len(set(curve.pcurve.weights)) != 1):
            return None
        count = overrides.get(edge.id)
        if count is None:
            count = max(1, round(edge_demand(source, edge.id, field)))
        if type(count) is not int or not 1 <= count <= max_divisions:
            return None
        parameters = [float(t) for t in _station_parameters(source, edge.id, count, field)]
        if len(parameters) != count+1 or parameters[0] != 0. or parameters[-1] != 1. or parameters != sorted(set(parameters)):
            return None
        stations = []
        for k, t in enumerate(parameters):
            p = tuple(curve.pcurve.exact_point(t))
            key = tuple(Fraction(x) for x in p)
            vertex = edge.start if k == 0 else edge.end if k == count else None
            # The owner's chord record binds the literal source vertex at an
            # endpoint and the curve's binary evaluation at an interior station.
            xyz = list(map(float, source.vertex_position(vertex) if vertex is not None else curve.evaluate(t)))
            row = [[float(x) for x in p], xyz]
            name = ('v', vertex) if vertex is not None else ('s', edge.id, k)
            if name not in local:
                if key in by_uv or key in fresh:
                    return None
                fresh.add(key)
                local[name] = len(rows)
                rows.append(row)
                exact_new.append(key)
            elif rows[local[name]] != row:
                return None
            stations.append([t, local[name]])
        rim.append([edge.id, stations])
        ordered = [n for _, n in stations]
        chains.append(ordered if use.forward else ordered[::-1])
    if (len({name for name in local if name[0] == 'v'}) != 4
            or any(a[-1] != b[0] for a,b in zip(chains, (*chains[1:], chains[0])))
            or len({n for c in chains for n in c[:-1]}) != sum(len(c)-1 for c in chains)):
        return None
    ring = [exact_new[n] for c in chains for n in c[:-1]]
    if _area(ring) < 0:
        chains = [c[::-1] for c in reversed(chains)]
    return chains, rim


def _compact_quality(rows, cells, settings):
    used = sorted({n for c in cells for n in c})
    index = {n:i for i,n in enumerate(used)}
    return _annulus_quality([rows[n] for n in used], [[index[n] for n in c] for c in cells], settings)


def _plan_multicell_annulus(source, data, base, by_uv, field, overrides, recombine, quality, check, limit):
    """Mesh a convex authored hole crossing internal edges of a regular Q4 grid.

    The owner station counts (m0, m1, m0, m1) define a W x H rectangle of whole
    parent cells containing every cell the hole touches. Its four outer sides
    and the four hole chains are matched corner to corner, one radial quad per
    station interval. Larger-than-minimal placements are tried in a fixed order.
    """
    from .surface_mesh import SurfaceMeshOptions
    face = source.faces[data['face']]
    cells, old = data['cells'], data['node_rows']
    if not recombine or len(face.holes[0]) != 4 or any(len(c) != 4 for c in cells):
        return None
    grid = _derive_grid(face, data, check)
    if grid is None:
        return None
    if quality is not None and not set(quality) <= {'max_element_growth','min_angle','max_angle',
                                                   'max_aspect_ratio','min_scaled_jacobian','max_warpage'}:
        return None
    g, cell_at = grid
    A, B = g.shape[0]-1, g.shape[1]-1
    rows = list(old)
    exact = [tuple(map(Fraction, row[0])) for row in old]
    got = _hole_chains(source, face, face.surface, field, overrides, by_uv, rows, exact, max(A, B))
    if got is None:
        return None
    chains, rim = got
    hull = [exact[c[0]] for c in chains]
    m = [len(c)-1 for c in chains]
    if not _convex(hull) or m[0] != m[2] or m[1] != m[3]:
        return None
    settings = SurfaceMeshOptions(**(quality or {}))
    uv = np.asarray([r[0] for r in old])
    corner_uv = np.asarray([rows[c[0]][0] for c in chains])
    lower, upper = corner_uv.min(axis=0), corner_uv.max(axis=0)
    touched = []
    for k, ((a,b), i) in enumerate(cell_at.items()):
        if k % 256 == 0:
            check_stage(check, 'source trim multicell touched material')
        box = uv[cells[i]]
        if np.any(box.max(axis=0) < lower) or np.any(box.min(axis=0) > upper):
            continue
        poly = [exact[n] for n in cells[i]]
        if not _convex(poly):
            return None
        if _interiors_meet(poly, hull):
            touched.append((a,b))
    if not touched:
        return None
    centre = tuple(sum(p[d] for p in hull)/4 for d in (0,1))
    min_a, max_a = min(a for a,_ in touched), max(a for a,_ in touched)
    min_b, max_b = min(b for _,b in touched), max(b for _,b in touched)
    shapes = [(m[0],m[1])] + ([(m[1],m[0])] if m[0] != m[1] else [])
    candidates = []
    for s,(W,H) in enumerate(shapes):
        if W > A or H > B:
            continue
        for a0 in range(max(0, max_a+1-W), min(min_a, A-W)+1):
            for b0 in range(max(0, max_b+1-H), min(min_b, B-H)+1):
                if len(candidates) % 256 == 0:
                    check_stage(check, 'source trim candidate rectangles')
                corners = [exact[int(g[a0+da*W, b0+db*H])] for da,db in ((0,0),(1,0),(1,1),(0,1))]
                mid = tuple(sum(p[d] for p in corners)/4 for d in (0,1))
                candidates.append((sum((x-y)**2 for x,y in zip(mid,centre)), s, a0, b0, W, H))
    for _, _, a0, b0, W, H in sorted(candidates):
        check_stage(check, 'source trim multicell footprint')
        sides = [[int(g[a,b0]) for a in range(a0,a0+W+1)], [int(g[a0+W,b]) for b in range(b0,b0+H+1)],
                 [int(g[a,b0+H]) for a in range(a0+W,a0-1,-1)], [int(g[a0,b]) for b in range(b0+H,b0-1,-1)]]
        if _area([exact[s[0]] for s in sides]) < 0:
            sides = [s[::-1] for s in reversed(sides)]
        lengths = [len(s)-1 for s in sides]
        options = []
        for r in range(4):
            if all(lengths[k] == m[(k+r)%4] for k in range(4)):
                inner = [chains[(k+r)%4] for k in range(4)]
                cost = sum(sum((x-y)**2 for x,y in zip(exact[s[0]], exact[c[0]])) for s,c in zip(sides, inner))
                options.append((cost, r, inner))
        selected = sorted(cell_at[a,b] for a in range(a0,a0+W) for b in range(b0,b0+H))
        if not options or any(not _convex([exact[n] for n in cells[i]]) for i in selected):
            continue
        interface = _perimeter([cells[i] for i in selected])
        if interface is None:
            continue
        removed = set(selected)
        kept = [c for i,c in enumerate(cells) if i not in removed]
        hole = [n for c in chains for n in c[:-1]]
        for _, _, inner in sorted(options, key=lambda o: o[:2]):
            proposed = [[o[i], o[i+1], n[i+1], n[i]] for o,n in zip(sides, inner) for i in range(len(o)-1)]
            check_stage(check, 'source trim multicell physical quality')
            if not (all(_convex([exact[n] for n in c]) for c in proposed)
                    and _exact_annulus_partition([cells[i] for i in selected], proposed, exact, interface, hole)
                    and _compact_quality(rows, kept+proposed, settings)):
                continue
            if _nominal(rows, proposed, limit):
                return dict(base, annulus=True, hole_cell_row=selected[0], hole_cell_interface_rows=list(interface),
                            new_rows=rows[len(old):], new_cells=proposed, parent_node_count=len(old),
                            replaced_cell_rows=selected, interface_rows=interface, hole_rim=rim,
                            quality=dict(quality or {}))
            expanded = _refine_ring(rows, exact, cells, set(selected), proposed, face.surface, hole,
                                    settings, limit, check)
            if expanded is not None:
                replaced, new_cells, new_rows, outer = expanded
                return dict(base, annulus=True, hole_cell_row=replaced[0], hole_cell_interface_rows=list(outer),
                            new_rows=new_rows[len(old):], new_cells=new_cells, parent_node_count=len(old),
                            replaced_cell_rows=replaced, interface_rows=outer, hole_rim=rim,
                            quality=dict(quality or {}))
    return None


def _plan_cut(cells, rim, hole, check):
    """Traverse actual oriented cell incidence, stopping at the new material rim."""
    incidence = defaultdict(list)
    for row, cell in enumerate(cells):
        if row % 256 == 0:
            check_stage(check, 'source trim parent incidence')
        for a,b in _sides(cell):
            incidence[_edge(a,b)].append((row,a,b))
    outer = {_edge(a[1], b[1]) for _,stations in rim for a,b in zip(stations,stations[1:])}
    cut = {_edge(a,b) for a,b in _sides(hole)}
    exposed = {edge for edge, adjacent in incidence.items() if len(adjacent) == 1}
    if outer != exposed or cut & outer or not cut <= incidence.keys():
        return None
    connections = defaultdict(list)
    for edge, adjacent in incidence.items():
        if len(adjacent) == 1:
            continue
        if len(adjacent) != 2:
            return None
        (i,a,b), (j,p,q) = adjacent
        if (a,b) != (q,p):
            return None
        if edge not in cut:
            connections[i].append(j)
            connections[j].append(i)
    def reachable(seed):
        seen = set(seed)
        pending = list(seen)
        visited = 0
        while pending:
            if visited % 256 == 0:
                check_stage(check, 'source trim owner cell partition')
            visited += 1
            row = pending.pop()
            for other in connections[row]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        return seen
    kept = reachable(incidence[e][0][0] for e in outer)
    removed = set(range(len(cells))) - kept
    if not removed or reachable((min(removed),)) != removed:
        return None
    for a,b in _sides(hole):
        # A clockwise hole has retained material to its left.
        if not any(i in kept and (p,q)==(a,b) for i,p,q in incidence[_edge(a,b)]):
            return None
    return sorted(removed)


def plan_source_face_trim(source, record, *, target_size, overrides=(), strategy='auto',
                          recombine=True, quality=None, cancellation_check=None):
    """Return a codec-safe fixed-perimeter trim plan or None for full-face generation.

    An aligned cut follows source-registered sides of whole parent cells. A
    convex four-sided hole inside one Q4 or crossing a rectangular grid of Q4s
    can instead receive a ring of new cells along its prescribed stations.
    Grading may include neighboring cells in the fixed-perimeter footprint.
    Ordinary generation certifies the station registry and the new material.
    """
    from anygeometry.nurbs import NurbsSurface, CurveOnSurface, NurbsCurve
    from anygeometry.nurbs import _IDENTITY
    from .hybrid import _station_parameters
    from .refinement import SizeField
    from .seeding import edge_demand
    data = record['data']
    if record['digest'] != _digest(data) or data['source'] != str(source.model_id):
        raise MeshError('invalid source trim parent proposal digest or namespace')
    face = data['face']
    definition = source.faces[face]
    support = definition.surface
    if (strategy not in ('auto','native',None) or type(support) is not NurbsSurface
            or len(definition.holes) != 1 or len(definition.holes[0]) < 3
            or not _ordinary_face(source, face, _vertex_faces(source), trim=True)
            or _face_signature(source, face, _native_definitions(source)[face], outer_only=True) != data['face_checksum']):
        return None
    cells = data['cells']
    rows = data['node_rows']
    if (not cells or any(len(c) not in (3,4) or len(set(c)) != len(c) or
            any(type(n) is not int or not 0 <= n < len(rows) for n in c) for c in cells)
            or {n for c in cells for n in c} != set(range(len(rows)))):
        raise MeshError('invalid source trim parent cell rows')
    if not recombine and any(len(c) == 4 for c in cells):
        return None
    uv = np.asarray([r[0] for r in rows], dtype=float)
    xyz = np.asarray([r[1] for r in rows], dtype=float)
    if (uv.shape != (len(rows),2) or xyz.shape != (len(rows),3) or not np.isfinite(uv).all()
            or not np.isfinite(xyz).all() or not np.all((0 <= uv) & (uv <= 1))):
        raise MeshError('invalid source trim parent coordinates')
    extent = float(np.linalg.norm(np.ptp(np.reshape(support.controls, (-1,3)), axis=0)))
    residual = source.tolerance.effective_surface_residual(extent)
    for start in range(0,len(rows),256):
        check_stage(cancellation_check, 'source trim parent support coordinates')
        chunk = slice(start,start+256)
        if np.any(np.linalg.norm(support.evaluate_many(uv[chunk])-xyz[chunk], axis=1) > residual):
            raise MeshError('source trim parent coordinates differ from current surface evaluation')
    by_uv = {tuple(map(Fraction,point)):i for i,point in enumerate(uv)}
    if len(by_uv) != len(rows):
        return None
    rim_rows = {n for _,stations in data['rim'] for _,n in stations}
    holes, rim = [], []
    field = SizeField(source, target_size)
    overrides = dict(overrides)
    if any(type(e) is not int or e not in source.edges or type(n) is not int or n < 1 for e,n in overrides.items()):
        return None
    base = {'face':face, 'proposal_digest':record['digest'], 'target_size':float(target_size),
            'strategy':strategy, 'overrides':[[e,n] for e,n in sorted(overrides.items())]}
    limit = float(target_size) + source.tolerance.effective_length(target_size)
    annulus = _plan_annulus(source,data,base,by_uv,field,overrides,recombine,quality,cancellation_check)
    if annulus is None:
        annulus = _plan_multicell_annulus(source,data,base,by_uv,field,overrides,recombine,quality,
                                          cancellation_check,limit)
    if annulus is not None:
        return annulus
    for use in definition.holes[0]:
        check_stage(cancellation_check, 'source trim owned hole interval')
        edge = source.edges[use.edge]
        curve = edge.curve
        if (type(curve) is not CurveOnSurface or curve.support != support or curve.transform != _IDENTITY
                or type(curve.pcurve) is not NurbsCurve or curve.pcurve.degree != 1
                or len(curve.pcurve.controls) != 2 or curve._iso_fixed() is None):
            return None
        fixed, _ = curve._iso_fixed()
        # Constant weights and one degree-one varying span make the world
        # trace an exact line. No hidden chord subdivision can be assumed.
        span = ((support.degree_v, len(support.controls[0])) if fixed == 0
                else (support.degree_u, len(support.controls)))
        if span != (1,2) or support.weights is not None and len({w for line in support.weights for w in line}) != 1:
            return None
        count = overrides.get(use.edge)
        if count is None:
            count = max(1, round(edge_demand(source, use.edge, field)))
        if type(count) is not int or not 1 <= count < len(rows):
            return None
        parameters = _station_parameters(source, use.edge, count, field)
        stations = [by_uv.get(tuple(curve.pcurve.exact_point(float(t)))) for t in parameters]
        if None in stations or len(set(stations)) != len(stations) or rim_rows.intersection(stations):
            return None
        if (not np.array_equal(rows[stations[0]][1], source.vertex_position(edge.start)) or
                not np.array_equal(rows[stations[-1]][1], source.vertex_position(edge.end))):
            return None
        if any(not np.array_equal(rows[n][1], tuple(float(x) for x in curve.exact_point(float(t))))
               for n,t in zip(stations, parameters)):
            return None
        rim.append([use.edge, [[float(t),node] for t,node in zip(parameters,stations)]])
        holes.append(stations if use.forward else list(reversed(stations)))
    if any(line[-1] != following[0] for line, following in zip(holes, (*holes[1:],holes[0]))):
        return None
    hole = [n for line in holes for n in line[:-1]]
    if len(set(hole)) != len(hole):
        return None
    polygon = [tuple(map(Fraction, rows[n][0])) for n in hole]
    area2 = sum(a[0]*b[1]-b[0]*a[1] for a,b in _sides(polygon))
    if not area2:
        return None
    if area2 > 0:
        hole.reverse()
    removed = _plan_cut(cells, data['rim'], hole, cancellation_check)
    if removed is None:
        return None
    return dict(base, hole_rim=rim, interface_rows=hole, replaced_cell_rows=removed)
