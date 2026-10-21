"""Bounded, transactional planar interior smoothing with fixed connectivity."""
from __future__ import annotations

from math import fsum, hypot, isfinite

import numpy as np

from ..errors import MeshError
from .domain import PlanarQuadDomain
from .quality_gate import corner_metrics, quad_violation, triangle_violation
from .state import QuadMeshState
from .validate import validate_planar_quad_result

_GROWTH_LIMIT = 1.5*(1.+.01)


# Unknown NumPy layouts and replaced callbacks retain the scalar route. These
# are the direct bindings used by norm's native one-dimensional float64 core.
_NORM = np.linalg.norm
_NORM_CORE = getattr(_NORM, "__wrapped__", None)
_NORM_GLOBALS = getattr(_NORM_CORE, "__globals__", {})
_NORM_SQRT = _NORM_GLOBALS.get("sqrt")
_NORM_ASARRAY = _NORM_GLOBALS.get("asarray")
_MEAN = np.mean
try:
    from numpy._core._multiarray_umath import _ArrayFunctionDispatcher
    from numpy._core.multiarray import asarray as _BUILTIN_ASARRAY
    from numpy._core.umath import sqrt as _BUILTIN_SQRT
    _NATIVE_NORM = (type(_NORM) is _ArrayFunctionDispatcher
                    and getattr(_NORM_CORE, "__name__", None) == "norm"
                    and _NORM_GLOBALS.get("__name__") == "numpy.linalg._linalg"
                    and type(_NORM_SQRT) is np.ufunc
                    and _NORM_SQRT is np.sqrt is _BUILTIN_SQRT
                    and _NORM_ASARRAY is _BUILTIN_ASARRAY
                    and type(_BUILTIN_ASARRAY).__name__ == "builtin_function_or_method")
    _NATIVE_MEAN = (type(_MEAN) is _ArrayFunctionDispatcher
                    and getattr(getattr(_MEAN, "__wrapped__", None), "__name__", None) == "mean")
except ImportError:  # Older supported NumPy versions use their scalar path.
    _NATIVE_NORM = False
    _NATIVE_MEAN = False


def _native_norm_available():
    return (_NATIVE_NORM and np.linalg.norm is _NORM
            and getattr(_NORM, "__wrapped__", None) is _NORM_CORE
            and _NORM_GLOBALS.get("sqrt") is _NORM_SQRT
            and _NORM_GLOBALS.get("asarray") is _NORM_ASARRAY
            and np.sqrt is _NORM_SQRT and np.asarray is _NORM_ASARRAY)


def _scalar_compatible_edge_norms(vectors):
    """Batch only single-component squares, retaining scalar dot for all others.

    Dot of a three-vector with <=1 nonzero component is its single square plus
    zeros. Restrict the exponent so square/sqrt cannot introduce ufunc warnings
    that dot does not emit. No replacement of an oblique dot by an axis sum.
    """
    native = _native_norm_available()
    if (not native or type(vectors) is not np.ndarray
            or vectors.dtype != np.dtype(np.float64) or vectors.ndim != 2
            or vectors.shape[1] != 3 or not vectors.flags.c_contiguous):
        return [np.linalg.norm(vector) for vector in vectors]
    # Callers bound chunks to 1024 edges. The helper itself also bounds working
    # arrays for direct/internal use without retaining any geometry references.
    values = []
    for start in range(0, len(vectors), 1024):
        block = vectors[start:start+1024]
        # Classify before abs/reductions: signalling NaNs must reach dot with
        # the original warning/exception behavior and row order.
        if not np.all(np.isfinite(block)):
            values.extend(np.linalg.norm(vector) for vector in block)
            continue
        magnitude = np.max(np.abs(block), axis=1)
        admitted = ((np.count_nonzero(block, axis=1) <= 1)
                    & ((magnitude == 0.)
                       | ((magnitude >= 2.**-500) & (magnitude <= 2.**500))))
        selected = magnitude[admitted]
        fast = iter(_NORM_SQRT(selected*selected))
        values.extend(next(fast) if use else np.linalg.norm(vector)
                      for use, vector in zip(admitted, block))
    return values


def _characteristic_lengths(xyz, index, cells):
    """Gather bounded native coordinates; keep per-cell mean arithmetic intact."""
    characteristic = {}
    native = (_native_norm_available() and _NATIVE_MEAN and np.mean is _MEAN
              and type(xyz) is np.ndarray and xyz.dtype == np.dtype(np.float64)
              and xyz.ndim == 2 and xyz.shape[1] == 3
              and type(cells) is dict and type(index) is dict
              and all(type(node) is int and type(row) is int for node, row in index.items())
              and all(type(cid) is int and type(body) is tuple
                      and len(body) in (3, 4)
                      and all(type(node) is int for node in body)
                      for cid, body in cells.items()))
    # These bounds make every gathered subtraction finite and normal (or zero).
    # Nondefault error policies therefore observe the same scalar norm failures.
    if native:
        for start in range(0, len(xyz), 1024):
            absolute = np.abs(xyz[start:start+1024])
            if not np.all((absolute == 0.)
                          | ((absolute >= 2.**-500) & (absolute <= 2.**500))):
                native = False
                break
    if not native:
        for cid, body in cells.items():
            characteristic[cid] = float(np.mean([np.linalg.norm(xyz[index[b]]-xyz[index[a]])
                for a, b in zip(body, body[1:]+body[:1])]))
        return characteristic
    from itertools import islice
    remaining = iter(cells.items())
    while block := tuple(islice(remaining, 256)):
        edge_indices = np.asarray([(index[a], index[b]) for _, body in block
                                   for a, b in zip(body, body[1:]+body[:1])], dtype=np.int64)
        differences = xyz[edge_indices[:, 1]]-xyz[edge_indices[:, 0]]
        lengths = _scalar_compatible_edge_norms(differences)
        offset = 0
        for cid, body in block:
            characteristic[cid] = float(np.mean(lengths[offset:offset+len(body)]))
            offset += len(body)
    return characteristic


def _common_evidence(positions, cells, pairs, domain, declared_edges=()):
    """Authoritative physical quality_v2 arithmetic for the final decision."""
    from ..core import MeshCore
    from ..quality_v2 import mesh_quality
    nodes = sorted(positions)
    index = {node: row for row, node in enumerate(nodes)}
    xyz = np.asarray([positions[node] if domain is None else domain.lift(positions[node])
                      for node in nodes], dtype=float)
    triangles = sorted(cid for cid in cells if len(cells[cid]) == 3)
    quads = sorted(cid for cid in cells if len(cells[cid]) == 4)
    core = MeshCore(xyz,
        np.asarray([[index[n] for n in cells[cid]] for cid in triangles], dtype=np.int64).reshape((-1, 3)),
        np.asarray([[index[n] for n in cells[cid]] for cid in quads], dtype=np.int64).reshape((-1, 4)),
        triangle_ids=np.asarray(triangles, dtype=np.int64), quad_ids=np.asarray(quads, dtype=np.int64))
    quality = mesh_quality(core, declared_plate_junction_edges=
                           tuple((index[a], index[b]) for a, b in declared_edges))
    result = {}
    for group in (quality.triangles, quality.quadrilaterals):
        for row, cid in enumerate(group.element_ids):
            result[("cell", int(cid))] = (
                max(0., float(group.maximum_angle[row])/150.-1.),
                max(0., 15./float(group.minimum_angle[row])-1.),
                max(0., float(group.aspect_ratio[row])/4.-1.),
                max(0., .2/float(group.scaled_jacobian[row])-1.),
                max(0., float(group.warpage[row])/.1-1.))
    characteristic = _characteristic_lengths(xyz, index, cells)
    for pair in pairs:
        a, b = (characteristic[cid] for cid in pair)
        result[("pair", pair)] = (max(0., max(a, b)/min(a, b)/_GROWTH_LIMIT-1.),)
    return result


def adjudicate_assembled_transition(mesh, original_positions, cancellation_check=None):
    """Pure full-shell veto, including adjacency across resident face boundaries."""
    if cancellation_check is not None:
        cancellation_check("quad-first:assembled-transition-audit")
    cells = {**mesh.quads, **mesh.tris}
    edges = {}
    for cid, body in cells.items():
        for a, b in zip(body, body[1:]+body[:1]):
            edges.setdefault(tuple(sorted((a, b))), []).append(cid)
    pairs = tuple(sorted({tuple(sorted(owners)) for owners in edges.values() if len(owners) == 2}))
    before_positions = dict(mesh.nodes)
    before_positions.update(original_positions)
    declared = mesh.declared_plate_junction_edges
    before = _common_evidence(before_positions, cells, pairs, None, declared)
    after = _common_evidence(mesh.nodes, cells, pairs, None, declared)
    initial = fsum(value for values in before.values() for value in values)
    final = fsum(value for values in after.values() for value in values)
    no_regression = all(new <= old for key in before for new, old in zip(after[key], before[key]))
    if cancellation_check is not None:
        cancellation_check("quad-first:assembled-transition-adjudication")
    return dict(accepted=no_regression and final < initial,
                status="ACCEPTED" if no_regression and final < initial else "ROLLED_BACK",
                initial_excess=initial, proposed_excess=final,
                cross_domain_pairs_included=True)


def smooth_planar_transitions(state, domain, *, budget, cancellation_check=None,
                              t3_minimum_angle=15., t3_angle_action="error"):
    """Only reduce existing common-policy excess; never create a new violation.

    Search lengths are geometric fractions of the shortest incident edge.
    All writes are staged in one transaction after the detached search succeeds.
    Original Quad gates, protected stations, IDs and cell incidence are fixed.
    """
    report = dict(status="DISABLED", evaluations=0, evaluation_budget=0,
                  moved_node_ids=[], initial_excess=0., final_excess=0., sweeps=0)
    if budget <= 0 or type(state) is not QuadMeshState or type(domain) is not PlanarQuadDomain:
        return report
    if cancellation_check is not None:
        cancellation_check("quad-first:planar-transition-setup")
    limit = min(int(budget), 8)*512
    report["evaluation_budget"] = limit
    cells = {int(cid): tuple(state.cell(cid)) for cid in state.cells}
    positions = {int(node): tuple(state.position(node)) for node in state.nodes}
    original = dict(positions)
    incident = {node: set() for node in positions}
    edges = {}
    for cid, body in cells.items():
        for node in body:
            incident[node].add(cid)
        for a, b in zip(body, body[1:]+body[:1]):
            edges.setdefault(tuple(sorted((a, b))), []).append(cid)
    protected = set(state.protected_nodes)
    for edge in state.protected_edges:
        protected.update(edge)
    for edge, owners in edges.items():
        if len(owners) != 2:
            protected.update(edge)
    pairs = tuple(sorted({tuple(sorted(owners)) for owners in edges.values() if len(owners) == 2}))
    pair_at = {cid: set() for cid in cells}
    for pair in pairs:
        for cid in pair:
            pair_at[cid].add(pair)

    def metrics(cid, changed=None):
        points = [changed[1] if changed is not None and node == changed[0] else positions[node]
                  for node in cells[cid]]
        minimum, maximum, jacobian, aspect = corner_metrics(points)
        if len(points) == 4:
            reason = quad_violation(points)
        else:
            reason = triangle_violation(points, t3_minimum_angle if t3_angle_action == "error" else 0.)
        if reason is not None:
            return None
        lengths = [hypot(b[0]-a[0], b[1]-a[1])
                   for a, b in zip(points, points[1:]+points[:1])]
        characteristic = fsum(lengths)/len(lengths)
        excess = (max(0., maximum/150.-1.), max(0., 15./minimum-1.),
                  max(0., aspect/4.-1.), max(0., .2/jacobian-1.))
        if not all(isfinite(value) for value in (*excess, characteristic)):
            return None
        return characteristic, excess

    cache = {cid: metrics(cid) for cid in cells}
    if any(value is None for value in cache.values()):
        report["status"] = "INELIGIBLE"
        return report

    def growth(pair, values):
        first, second = (values[cid][0] for cid in pair)
        return max(0., max(first, second)/min(first, second)/_GROWTH_LIMIT-1.)

    pair_cache = {pair: growth(pair, cache) for pair in pairs}
    initial = fsum(value for _, excess in cache.values() for value in excess) + fsum(pair_cache.values())
    report["initial_excess"] = report["final_excess"] = initial
    if initial == 0.:
        report["status"] = "NOIMPROVE"
        return report
    tx = state.transaction()  # captures source generation before callbacks
    moved = set()
    directions = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.),
                  (1., 1.), (1., -1.), (-1., 1.), (-1., -1.))
    for sweep in range(8):
        report["sweeps"] = sweep+1
        bad = {cid for cid, (_, excess) in cache.items() if any(excess)}
        for pair, excess in pair_cache.items():
            if excess:
                bad.update(pair)
        candidates = sorted({node for cid in bad for node in cells[cid]}-protected)
        progress = False
        for node in candidates:
            if report["evaluations"] >= limit:
                break
            if cancellation_check is not None:
                cancellation_check("quad-first:planar-transition-search")
            affected = tuple(sorted(incident[node]))
            local_pairs = tuple(sorted({pair for cid in affected for pair in pair_at[cid]}))
            before = fsum(value for cid in affected for value in cache[cid][1]) + fsum(pair_cache[pair] for pair in local_pairs)
            if before == 0.:
                continue
            here = positions[node]
            neighbors = sorted({other for cid in affected for other in cells[cid]}-{node})
            scale = min(hypot(positions[other][0]-here[0], positions[other][1]-here[1]) for other in neighbors)
            best = None
            for divisor in (8., 16., 32., 64., 128.):
                for dx, dy in directions:
                    if report["evaluations"] >= limit:
                        break
                    point = (here[0]+dx*scale/divisor, here[1]+dy*scale/divisor)
                    report["evaluations"] += 1
                    values = {cid: metrics(cid, (node, point)) for cid in affected}
                    if any(value is None for value in values.values()):
                        continue
                    # Every previously passing constraint remains passing, and
                    # no already-failing constraint becomes worse.
                    if any(new > old for cid in affected
                           for new, old in zip(values[cid][1], cache[cid][1])):
                        continue
                    merged = {cid: values.get(cid, cache[cid]) for pair in local_pairs for cid in pair}
                    changes = {pair: growth(pair, merged) for pair in local_pairs}
                    if any(value > pair_cache[pair] for pair, value in changes.items()):
                        continue
                    score = fsum(value for _, excess in values.values() for value in excess) + fsum(changes.values())
                    if score < before and (best is None or score < best[0]):
                        best = score, point, values, changes
            if best is not None:
                _, positions[node], values, changes = best
                cache.update(values)
                pair_cache.update(changes)
                moved.add(node)
                progress = True
        if not progress or report["evaluations"] >= limit:
            break
    if not moved:
        tx.rollback()
        report["status"] = "NOIMPROVE"
        return report
    # The cheap chart search is only a proposal. Physical quality_v2 and the
    # exact common mean-edge arithmetic decide whether it may be published.
    before = _common_evidence(original, cells, pairs, domain)
    after = _common_evidence(positions, cells, pairs, domain)
    if (any(new > old for key in before for new, old in zip(after[key], before[key]))
            or not fsum(value for values in after.values() for value in values)
            < fsum(value for values in before.values() for value in values)):
        tx.rollback()
        report["status"] = "REJECTED_COMMON_QUALITY"
        return report
    for node in sorted(moved):
        tx.move_node(node, positions[node])
    try:
        validate_planar_quad_result(tx.view, t3_minimum_angle=t3_minimum_angle,
                                   t3_angle_action=t3_angle_action)
    except MeshError:
        tx.rollback()
        report["status"] = "REJECTED_VALIDATION"
        return report
    if cancellation_check is not None:
        cancellation_check("quad-first:planar-transition-commit")
    tx.commit()
    report.update(status="APPLIED", moved_node_ids=sorted(moved),
                  _original_positions=tuple((node, original[node]) for node in sorted(moved)),
                  final_excess=fsum(value for _, excess in cache.values() for value in excess)+fsum(pair_cache.values()),
                  max_displacement=max(hypot(positions[n][0]-original[n][0], positions[n][1]-original[n][1]) for n in moved))
    return report
