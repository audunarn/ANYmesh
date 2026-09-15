"""Scope and evidence transport tests that never generate a mesh."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def measure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("performance_dev_measure")


def test_direct_worker_rejects_excluded_work(measure):
    base = ["--case", "planar_callback_absent", "--scale", "500k",
            "--requested-elements", "500000", "--route", "legacy",
            "--backend", "python", "--install-kind", "source",
            "--source-commit", "a" * 40, "--compiler-id", "none",
            "--output", "unused.json"]
    with pytest.raises(ValueError, match="EXCLUDED_BY_SCOPE"):
        measure._arguments(base)


def test_raw_sample_receipts_are_create_new(measure, tmp_path):
    path = tmp_path / "sample.json"
    measure._write_new(path, {"ordinal": 1})
    with pytest.raises(FileExistsError):
        measure._write_new(path, {"ordinal": 2})


def test_missing_frontal_work_remains_unknown(measure):
    assert measure._work({}, "frontal")["published_insertions"] is None
    assert measure._work({}, "legacy")["published_insertions"] == 0


def test_callback_and_work_rows_are_not_fabricated(measure):
    diag = {"faces": [{"accepted_insertions": 3}, {"accepted_insertions": 2}]}
    assert measure._work(diag, "frontal")["published_insertions"] == 5
    staged = {"native_v2": {"insertions": 500, "published_insertions": 0}}
    assert measure._work(staged, "frontal")["insertions"] == 500
    assert measure._work(staged, "frontal")["published_insertions"] == 0
    duplicated = {"triangulation_backend_by_face": {"1": {
        "native_v2": {"insertions": 55, "published_insertions": 55,
                      "topology_operations": 222},
        "native_diagnostics": {"native_v2": {"insertions": 55,
                                              "topology_operations": 222}},
    }}}
    assert measure._work(duplicated, "frontal") == {
        "insertions": 55, "published_insertions": 55, "topology_operations": 222,
    }
    duplicated["triangulation_backend_by_face"]["1"]["native_v2"].pop("insertions")
    assert measure._work(duplicated, "frontal")["insertions"] is None


def test_association_digest_tracks_owner_identity(measure):
    class Snapshot:
        nodes_of_edge = {2: (3, 4)}
        elements_of_face = {7: (9,)}
        node_of_vertex = {1: 3}
        declared_plate_junction_edges = ((3, 4),)

    first = measure._association_digest(Snapshot())
    changed = Snapshot()
    changed.elements_of_face = {8: (9,)}
    assert measure._association_digest(changed) != first


def test_source_version_comes_from_source_metadata(measure, tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ANYmesher"\nversion = "0.5.0"\n', encoding="ascii"
    )
    assert measure._source_version(tmp_path) == "0.5.0"
