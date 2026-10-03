"""Detached, interior-only two-flip closures in an existing topology budget."""
from copy import deepcopy

from .native_v2 import MutableT3Topology, _GeometryLimited


def _detached(topology):
    # Keep row/owner correspondence rather than sorting connectivity alone.
    trial = MutableT3Topology(topology._points, topology._triangles,
        tuple(topology.protected_edges), node_owners=topology.node_owners,
        triangle_owners=topology.triangle_owners,
        splittable_edges=topology._splittable_intervals,
        seed_registry=topology._seed_registry)
    trial._shared_node_ids = dict(topology.shared_node_ids)
    trial.epoch = topology.epoch
    trial.free_triangle_ids = list(topology.free_triangle_ids)
    trial.quality_cache = deepcopy(topology.quality_cache)
    return trial


def two_flip_cavity(topology, baseline, first_edges, *, score, progress,
                    owner_checkpoint, charge_attempt, stats,
                    cancellation_check=None):
    """Return a complete detached proposal; never mutate the committed topology.

    ``charge_attempt(kind)`` consumes the caller's existing allowance and updates
    its counters BEFORE any attempted primitive/cancellation callback. False
    means no allowance remains. The first flip permits valid intermediate poor
    quality; every complete pair must pass ``progress`` against ``baseline``.
    Second flips are limited to the first quad's perimeter (at most four edges),
    excluding its new diagonal. Thus this is a connected two-edit cavity, not a
    second independent global flip pass. Flips never resolve a registry station.
    """
    def check(phase):
        if cancellation_check is not None:
            cancellation_check(phase)

    def eligible(state, edge):
        return (edge not in state.protected_edges and edge not in state.splittable_edges
                and len(state._topology_index.attached(edge)) == 2)

    def attempt(state, edge, kind):
        if not charge_attempt(kind):
            stats['stop_reason'] = 'topology_budget'
            return None, None
        check('native-v2 physical cavity ' + kind + ' attempt')
        staged = None
        refusal = None

        def validate(points, cells):
            nonlocal staged, refusal
            proposal = score(points, cells)
            if proposal.report['invalid_element_count'] or (
                    kind == 'second' and not progress(baseline.report, proposal.report)):
                refusal = _GeometryLimited('physical cavity makes no admissible complete progress'
                    if kind == 'second' else 'physical cavity intermediate is invalid')
                raise refusal
            staged = proposal

        try:
            changed = state.flip_edge(edge, cancellation_check=cancellation_check,
                                      _validate_candidate=validate)
        except _GeometryLimited as error:
            # Only our local guard refusal is recoverable. Owner/cancel/scorer
            # exceptions retain their exact identity, even of the same type.
            if error is not refusal:
                raise
            changed = False
        if not changed:
            stats[kind + '_refusals'] += 1
        return changed, staged

    owner_checkpoint()
    seen = set()
    for raw_edge in first_edges:
        check('native-v2 physical cavity first candidate scan')
        edge = tuple(sorted(map(int, raw_edge)))
        if edge in seen or not eligible(topology, edge):
            continue
        seen.add(edge)
        # The perimeter connects the original two cells to at most one outside
        # cell on the second edit. Its constraints are excluded, never split.
        attached = topology._topology_index.attached(edge)
        perimeter = sorted({tuple(sorted((int(cell[i]), int(cell[(i+1) % 3]))))
            for cell in topology._triangles[list(attached)] for i in range(3)} - {edge})
        first = _detached(topology)
        changed, _ = attempt(first, edge, 'first')
        if changed is None:
            return None
        if not changed:
            continue
        for second_edge in perimeter:
            check('native-v2 physical cavity second candidate scan')
            if not eligible(first, second_edge):
                continue
            second = _detached(first)
            changed, proposal = attempt(second, second_edge, 'second')
            if changed is None:
                return None
            if not changed:
                continue
            # Detached topology and matching full cached report are adopted
            # together only after current owner binding and cancellation pass.
            check('native-v2 physical cavity complete admission')
            owner_checkpoint()
            stats['commits'] += 1
            stats['stop_reason'] = 'committed'
            return second, proposal
    owner_checkpoint()
    stats['stop_reason'] = 'no_progress_for_local_two_flip_cavities'
    return None
