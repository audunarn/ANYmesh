import numpy as np
import pytest

from anymesher.native_v2 import _metric_edge_pairs


@pytest.mark.parametrize("refused", ((1, 2), (3, 4)))
def test_metric_lengths_follow_filtered_splittable_edges(refused):
    original = ((1, 2), (3, 4), (5, 6))
    filtered = np.asarray([edge for edge in original if edge != refused], dtype=np.int64)
    lengths = np.asarray((1.25, 2.5), dtype=np.float64)
    assert _metric_edge_pairs(filtered, lengths) == (
        (tuple(filtered[0]), 1.25),
        (tuple(filtered[1]), 2.5),
    )


def test_metric_edge_pair_count_mismatch_fails_closed():
    with pytest.raises(Exception, match="metric rows are misaligned"):
        _metric_edge_pairs(((1, 2), (3, 4)), (1.25,))
