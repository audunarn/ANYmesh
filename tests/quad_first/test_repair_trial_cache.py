"""Repair caches require exact inputs and cannot retain mutable proof state."""
import numpy as np
from anymesher.hybrid import _RepairTrialCache


def test_trial_cache_detaches_inputs_and_outputs_and_detects_moved_midsides():
    cache = _RepairTrialCache(8)
    nodes = np.arange(24., dtype=float).reshape(8,3)
    original = nodes.copy()
    normal = np.array([0.,0.,1.])
    key = ('face', 'family', b'samples', 'revision')
    cache.put(key, nodes, .5, normal)
    assert cache.get(key, nodes)[0] == .5
    nodes[-1, 1] += .01
    assert cache.get(key, nodes) is None
    assert cache.get(key, original.astype(np.float32)) is None
    nodes[:] = original
    normal[:] = 99.
    retrieved = cache.get(key, nodes)[1]
    np.testing.assert_array_equal(retrieved, [0.,0.,1.])
    retrieved[:] = -99.
    np.testing.assert_array_equal(cache.get(key, nodes)[1], [0.,0.,1.])
    assert cache.get(('other samples',), nodes) is None
    assert cache.get(('other revision',), nodes) is None


def test_trial_cache_is_bounded_and_replaces_changed_cell_binding():
    cache = _RepairTrialCache(2)
    nodes = np.zeros((8,3)); normal = np.array([0.,0.,1.])
    for key in range(5):
        cache.put(key,nodes,float(key),normal)
        assert len(cache.entries) <= 2
    assert cache.get(0,nodes) is None
    changed = nodes.copy(); changed[0,0] = 1.
    cache.put(4,changed,.75,normal)
    assert cache.get(4,nodes) is None
    assert cache.get(4,changed)[0] == .75
