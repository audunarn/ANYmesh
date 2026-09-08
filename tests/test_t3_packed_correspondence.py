"""Lossless packed row correspondence versus the general reference."""
import numpy as np
import pytest

from anymesher import _t3_insertion_result as result
from anymesher import _t3_packed_correspondence as packed


@pytest.mark.parametrize("maximum", [0, 1, 2, 3, 1023, 1024, 1048576, 2097151])
def test_bounded_keys_match_reference_at_bit_boundaries(maximum):
    rng = np.random.default_rng(91827)
    for count in (0, 1, 7, 128):
        before = rng.integers(0, maximum + 1, (count, 3), dtype=np.int64)
        before = np.asarray(sorted(map(tuple, before)), dtype=np.int64).reshape(-1, 3)
        after = rng.integers(0, maximum + 1, (count + 9, 3), dtype=np.int64)
        if count:
            after[:min(5, count)] = before[:min(5, count)]
        old_bytes, new_bytes = before.tobytes(), after.tobytes()
        expected = result._reference_source_rows(before, after)
        actual = packed.packed_source_rows(before, after)
        assert actual is not None
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(result._compute_source_rows(before, after), expected)
        assert before.tobytes() == old_bytes and after.tobytes() == new_bytes


def test_duplicate_rows_keep_the_first_original_source_and_orientation():
    before = np.array(((0, 1, 2), (0, 1, 2), (0, 2, 1), (3, 4, 5)), dtype=np.int64)
    after = np.array(((3, 4, 5), (0, 1, 2), (1, 2, 0),
                      (0, 2, 1), (0, 1, 2)), dtype=np.int64)
    expected = np.array((3, 0, -1, 2, 0))
    np.testing.assert_array_equal(packed.packed_source_rows(before, after), expected)
    np.testing.assert_array_equal(result._reference_source_rows(before, after), expected)


def test_canonical_path_uses_only_unsigned_integer_search(monkeypatch):
    before = np.array(((0, 1, 2), (0, 2, 3), (2, 4, 3)), dtype=np.int64)
    after = np.array(((0, 1, 5), (2, 4, 3), (0, 2, 3)), dtype=np.int64)
    expected = result._reference_source_rows(before, after)
    search = np.searchsorted
    calls = []
    def tracked(first, second, *args, **kwargs):
        assert first.dtype == second.dtype == np.dtype(np.uint64)
        calls.append(len(first))
        return search(first, second, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("structured keys or sorting reached canonical path")
    monkeypatch.setattr(np, "searchsorted", tracked)
    monkeypatch.setattr(np, "argsort", forbidden)
    monkeypatch.setattr(result, "_keys", forbidden)
    np.testing.assert_array_equal(result._compute_source_rows(before, after), expected)
    assert calls == [len(before)]


@pytest.mark.parametrize("case", ["unordered", "negative", "wide", "int64_max",
                                 "int32", "float64", "non_native_endian"])
def test_general_domains_retain_reference_semantics(monkeypatch, case):
    before = np.array(((0, 1, 2), (0, 2, 3), (2, 4, 3)), dtype=np.int64)
    after = before[[2, 0]].copy()
    if case == "unordered":
        before = before[::-1].copy()
    elif case == "negative":
        before, after = before - 5, after - 5
    elif case == "wide":
        before, after = before + 2**21, after + 2**21
    elif case == "int64_max":
        before = before + (np.iinfo(np.int64).max - 4)
        after = after + (np.iinfo(np.int64).max - 4)
    elif case in ("int32", "float64"):
        before, after = before.astype(case), after.astype(case)
    else:
        endian = ">i8" if np.little_endian else "<i8"
        before, after = before.astype(endian), after.astype(endian)
    expected = result._reference_source_rows(before, after)
    assert packed.packed_source_rows(before, after) is None
    calls = []
    reference = result._reference_source_rows
    def tracked(old, new):
        calls.append(True)
        return reference(old, new)
    monkeypatch.setattr(result, "_reference_source_rows", tracked)
    np.testing.assert_array_equal(result._compute_source_rows(before, after), expected)
    assert calls == [True]


@pytest.mark.parametrize("readonly", [False, True])
def test_strided_storage_and_empty_queries(readonly):
    backing = np.zeros((12, 6), dtype=np.int64)
    before = backing[::2, ::2]
    before[:] = np.arange(18).reshape(-1, 3)
    after = before[::-2]
    if readonly:
        before.setflags(write=False)
        after.setflags(write=False)
    np.testing.assert_array_equal(
        packed.packed_source_rows(before, after),
        result._reference_source_rows(before, after))
    empty = np.empty((0, 3), dtype=np.int64)
    np.testing.assert_array_equal(packed.packed_source_rows(before, empty), [])
    np.testing.assert_array_equal(packed.packed_source_rows(empty, after),
                                  np.full(len(after), -1))


def test_packing_is_injective_and_preserves_lexicographic_order():
    cells = np.array([(a, b, c) for a in range(8)
                      for b in range(8) for c in range(8)], dtype=np.int64)
    encoded = packed._encode(cells, 3)
    assert len(np.unique(encoded)) == len(cells)
    assert np.all(encoded[1:] > encoded[:-1])
