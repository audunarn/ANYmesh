"""Bounded alternating-path repair of a detached physical T3/Q4 matching."""
import numpy as np

def _matching_operation_pin(function):
    wrapped=getattr(function,'__wrapped__',function)
    return (function,getattr(function,'__code__',None),repr(getattr(function,'__defaults__',None)),
        repr(getattr(function,'__kwdefaults__',None)),wrapped,getattr(wrapped,'__code__',None),
        repr(getattr(wrapped,'__defaults__',None)),repr(getattr(wrapped,'__kwdefaults__',None)))


_MATCHING_NUMPY=tuple(_matching_operation_pin(f) for f in (np.asarray,np.arange,np.zeros,np.unique,np.minimum))


def _matching_numpy_current():
    return _MATCHING_NUMPY==tuple(_matching_operation_pin(f) for f in (np.asarray,np.arange,np.zeros,np.unique,np.minimum))



def repair_matching(mesh, candidates, selected, *, min_scaled_jacobian,
                    max_aspect_ratio, min_angle, max_angle, max_warpage,
                    max_growth, work_limit, cancellation_check=None):
    """Improve complete matchings without changing any geometric/topological row.

    Every candidate quad has already passed the caller's shape gates. Original
    triangles remain available; alternating paths only change which of them are
    paired. Intermediate search states are detached and never published.
    """
    from .quality_v2 import evaluate_quality
    quality = evaluate_quality(mesh).triangles
    active = np.asarray(mesh.triangle_active, dtype=bool)
    poor = active & ((quality.scaled_jacobian < min_scaled_jacobian)
        | (quality.aspect_ratio > max_aspect_ratio)
        | (quality.minimum_angle < min_angle)
        | (quality.maximum_angle > max_angle)
        | (quality.warpage > max_warpage))
    rows = mesh.triangle_connectivity[:, :3]
    xyz = mesh.node_coordinates[rows]
    triangle_lengths = np.linalg.norm(np.roll(xyz, -1, axis=1)-xyz, axis=2).mean(axis=1)
    quad_lengths = []
    options = {}
    for index, candidate in enumerate(candidates):
        points = mesh.node_coordinates[list(candidate.corners)]
        quad_lengths.append(float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).mean()))
        options.setdefault(candidate.first, []).append(index)
        options.setdefault(candidate.second, []).append(index)
    # Candidates are already shape-qualified. Indexing merely replaces many
    # tiny writes; matching overlap keeps the exact sequential-write fallback.
    from .recombine import _Candidate
    known=(all(type(c) is _Candidate for c in candidates)
           and _matching_numpy_current())
    candidate_pairs=tuple((c.first,c.second) for c in candidates) if known else None
    indexed=np.asarray(candidate_pairs,dtype=np.int64).reshape((-1,2)) if known else None
    candidate_labels=None if indexed is None else np.minimum(indexed[:,0],indexed[:,1])
    indexed_lengths=None if indexed is None else np.asarray(quad_lengths)
    candidate_index = {(c.first, c.second): i for i, c in enumerate(candidates)}
    current = frozenset(candidate_index[(c.first, c.second)] for c in selected)
    incidence = {}
    for row in np.flatnonzero(active):
        for a, b in zip(rows[row], np.roll(rows[row], -1)):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(row))
    neighbours = np.asarray([owners for owners in incidence.values() if len(owners) == 2], dtype=int).reshape((-1, 2))
    used = 0
    # A connected cavity is deliberately smaller than the global matching
    # budget. This is a search-family limit, never additional work authority.
    trial_limit = min(work_limit, 4096)

    def check():
        nonlocal indexed
        if cancellation_check is not None:
            cancellation_check('physical recombination alternating path')
            if indexed is not None and tuple((c.first,c.second) for c in candidates)!=candidate_pairs:
                indexed=None  # retain dynamic candidate reads after external mutation

    def score(state):
        paired = np.zeros(len(rows), dtype=bool)
        labels = np.arange(len(rows))
        lengths = triangle_lengths.copy()
        partners = {}
        fast=(indexed is not None and _matching_numpy_current())
        indices=np.asarray(tuple(state),dtype=np.int64) if fast else None
        ends=indexed[indices] if fast else None
        fast=fast and len(np.unique(ends))==2*len(indices)
        if fast:
            paired[ends]=True
            labels[ends]=candidate_labels[indices,None]
            lengths[ends]=indexed_lengths[indices,None]
        for index in state:
            c = candidates[index]
            if not fast:
                paired[[c.first, c.second]] = True
                labels[[c.first, c.second]] = min(c.first, c.second)
                lengths[[c.first, c.second]] = quad_lengths[index]
            partners[c.first] = partners[c.second] = index
        exposed = neighbours[labels[neighbours[:, 0]] != labels[neighbours[:, 1]]]
        if len(exposed):
            a, b = lengths[exposed[:, 0]], lengths[exposed[:, 1]]
            ratios = np.maximum(a/b, b/a)
            failed = ratios > max_growth
            growth_rows = set(map(int, exposed[failed].ravel()))
            excess = float(np.maximum(ratios-max_growth, 0.).sum())
            maximum = float(np.max(ratios, initial=1.))
        else:
            failed = np.empty(0, dtype=bool)
            growth_rows, excess, maximum = set(), 0., 1.
        bad = set(map(int, np.flatnonzero(poor & ~paired)))
        # Scientific acceptance is unchanged. This ordering is a search
        # heuristic; final independent mixed-cell checks still govern admission.
        key = (len(bad), int(np.count_nonzero(failed)), excess, maximum, -len(state))
        return key, sorted(bad | growth_rows), partners

    key, roots, partners = score(current)
    commits = 0
    while roots and used < trial_limit:
        check()
        best, best_key = None, key
        seen = {current}

        def search(state, free, depth, visited):
            nonlocal used, best, best_key
            if depth == 0 or used >= trial_limit:
                return
            for index in options.get(free, ()):
                if used >= trial_limit:
                    return
                check()
                c = candidates[index]
                other = c.second if free == c.first else c.first
                if other in visited:
                    continue
                occupied = next((j for j in state if other in (candidates[j].first, candidates[j].second)), None)
                proposal = (state - ({occupied} if occupied is not None else set())) | {index}
                if proposal in seen:
                    continue
                seen.add(proposal)
                used += 1
                proposed_key, _, _ = score(proposal)
                if proposed_key < best_key:
                    best, best_key = proposal, proposed_key
                if occupied is not None:
                    old = candidates[occupied]
                    following = old.second if other == old.first else old.first
                    search(proposal, following, depth-1, visited | {free, other})

        for root in roots:
            if used >= trial_limit:
                break
            occupied = partners.get(root)
            detached = current - ({occupied} if occupied is not None else set())
            free_rows = [root]
            if occupied is not None:
                c = candidates[occupied]
                free_rows.append(c.second if root == c.first else c.first)
            for free in free_rows:
                search(detached, free, 4, set())
        if best is None:
            break
        check()
        current, key = frozenset(best), best_key
        key, roots, partners = score(current)
        commits += 1
    check()
    return [candidates[i] for i in sorted(current)], used, commits, bool(roots and used >= trial_limit)
