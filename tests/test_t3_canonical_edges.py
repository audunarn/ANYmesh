"""Differential canonical-edge snapshots after local incidence changes."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from anymesher import _t3_incidence as incidence
from anymesher._t3_incidence import T3IncidenceIndex


def oracle(triangles):
    edges = {tuple(sorted((int(a), int(b))))
             for row in triangles for a, b in zip(row, (*row[1:], row[0]))}
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def grid(flips, width=4):
    rows = []
    for y in range(width):
        for x in range(width):
            a = y * (width + 1) + x
            b, d = a + 1, a + width + 1
            c = d + 1
            rows.extend(((a, b, d), (b, c, d)) if y * width + x in flips
                        else ((a, b, c), (a, c, d)))
    return np.asarray(rows, dtype=np.int64)


def test_local_flips_and_row_reordering_match_full_scan_exactly():
    cells = grid(set())
    first = index = T3IncidenceIndex(cells)
    first_bytes = first.canonical_edges().tobytes()
    flips = set()
    rng = np.random.default_rng(941)
    for iteration in range(40):
        flips.symmetric_difference_update(rng.choice(16, size=3, replace=False).tolist())
        cells = grid(flips)
        if iteration % 2:
            cells = cells[::-1]
        old = index
        old_bytes = old.canonical_edges().tobytes()
        index = index.updated(cells)
        assert index.canonical_edges().tobytes() == oracle(cells).tobytes()
        assert old.canonical_edges().tobytes() == old_bytes
        for edge in map(tuple, index.canonical_edges()):
            expected = tuple(i for i, row in enumerate(cells) if set(edge).issubset(row))
            assert index.attached(edge) == expected
    assert first.canonical_edges().tobytes() == first_bytes


def test_unchanged_edge_set_shares_internal_array_but_exports_are_defensive():
    cells = grid(set())
    index = T3IncidenceIndex(cells)
    reordered = index.updated(cells[::-1])
    assert reordered._edge_rows is index._edge_rows
    assert not index._edge_rows.flags.writeable
    exported = index.canonical_edges()
    exported[:] = -1
    np.testing.assert_array_equal(index.canonical_edges(), oracle(cells))
    np.testing.assert_array_equal(reordered.canonical_edges(), oracle(cells))


def test_export_does_not_reconstruct_membership(monkeypatch):
    cells = grid(set())
    index = T3IncidenceIndex(cells)
    def forbidden(_):
        pytest.fail("canonical export reconstructed triangle edges")
    monkeypatch.setattr(incidence, "_edges", forbidden)
    np.testing.assert_array_equal(index.canonical_edges(), oracle(cells))


def test_only_changed_cells_recompute_membership(monkeypatch):
    before, after = grid(set()), grid({5})
    index = T3IncidenceIndex(before)
    original = incidence._edges
    calls = []
    def observed(cell):
        calls.append(cell)
        return original(cell)
    monkeypatch.setattr(incidence, "_edges", observed)
    updated = index.updated(after)
    assert len(calls) == 4  # Two removed and two added triangles, not all 32.
    np.testing.assert_array_equal(updated.canonical_edges(), oracle(after))


@pytest.mark.parametrize("phase", ("native-v2 topology-edge scan", "native-v2 native incidence commit"))
def test_cancelled_export_or_update_keeps_source_snapshot(phase):
    index = T3IncidenceIndex(grid(set()))
    expected = index.canonical_edges().tobytes()
    error = RuntimeError("cancel edge snapshot")
    def cancel(stage):
        if stage == phase:
            raise error
    with pytest.raises(RuntimeError) as caught:
        if phase == "native-v2 topology-edge scan":
            index.canonical_edges(cancellation_check=cancel)
        else:
            index.updated(grid({2}), cancellation_check=cancel)
    assert caught.value is error
    assert index.canonical_edges().tobytes() == expected


def test_deletion_to_empty_then_addition_is_canonical():
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    first = T3IncidenceIndex(cells)
    empty = first.updated(np.empty((0, 3), dtype=np.int64))
    assert empty.canonical_edges().shape == (0, 2)
    added = empty.updated(cells)
    np.testing.assert_array_equal(added.canonical_edges(), oracle(cells))


def test_cylindrical_refinement_matches_full_edge_reconstruction(monkeypatch):
    from anymesher.hybrid import generate_hybrid_mesh_result, _neutral_shell_core
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    bench = importlib.import_module("native_v2_baseline")
    fixture = bench.cylinder_case("cylinder_patch")
    def generate():
        return generate_hybrid_mesh_result(
            fixture.model, face_ids=fixture.face_ids, target_size=.4, strategy="native",
            overrides=fixture.overrides(.4), refinements=fixture.refinements(.4),
            native_backend="python", native_options=fixture.options(.4, "frontal", 128),
            recombine=True,
        )
    incremental = generate()
    calls = []
    def reference(self, *, cancellation_check=None):
        calls.append(len(self._row_by_cell))
        return oracle(tuple(self._row_by_cell))
    monkeypatch.setattr(T3IncidenceIndex, "canonical_edges", reference)
    full_scan = generate()
    assert calls
    assert bench._mesh_digest(_neutral_shell_core(incremental.mesh)) == (
        bench._mesh_digest(_neutral_shell_core(full_scan.mesh)))
    assert incremental.triangulation_backend_by_face[1]["native_v2"] == (
        full_scan.triangulation_backend_by_face[1]["native_v2"])
