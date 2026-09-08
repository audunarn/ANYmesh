"""Bounded behavioral checks for work-bearing native scaling evidence."""
from copy import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("native_v2_baseline")


def work(insertions=3, operations=7):
    return {
        "faces": {"1": {"insertions": insertions, "topology_operations": operations,
                        "published_insertions": insertions,
                        "selected_route": "frontal_delaunay", "cancelled": False}},
        "insertions": insertions, "published_insertions": insertions,
        "topology_operations": operations,
    }


def record():
    return {
        "schema": "anymesher.native-v2-baseline/3",
        "case": "cylinder_patch", "route": "frontal", "scale": "10k",
        "source_commit": "a" * 40, "requested_elements": 10000,
        "actual_elements": 10000, "element_count_ratio": 1.,
        "warmups": 1, "repetitions": 7, "repetition_digests": ["b" * 64] * 7,
        "native_work_samples": [work() for _ in range(7)],
        "durations_seconds": [1., 2., 3., 4., 5., 6., 7.],
        "median_seconds": 4., "peak_rss_bytes": 1000,
        "quality": {"minimum_scaled_jacobian": .5},
    }


def test_real_work_record_is_accepted(benchmark, tmp_path):
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    assert benchmark._load_record(path)["native_work_samples"][0]["insertions"] == 3


@pytest.mark.parametrize("change", (
    lambda row: row.update(schema="anymesher.native-v2-baseline/2"),
    lambda row: row.pop("native_work_samples"),
    lambda row: row.update(native_work_samples=[work(0, 0) for _ in range(7)]),
    lambda row: row["native_work_samples"][0].update(insertions=99),
    lambda row: row["native_work_samples"][0]["faces"]["1"].update(insertions=True),
    lambda row: row["native_work_samples"][0]["faces"]["1"].update(selected_route="mapped"),
    lambda row: row["native_work_samples"].__setitem__(0, work(4, 8)),
    lambda row: row.update(median_seconds=.1),
    lambda row: row["durations_seconds"].__setitem__(0, float("nan")),
    lambda row: row["durations_seconds"].pop(),
    lambda row: row.update(peak_rss_bytes=None),
    lambda row: row["quality"].update(minimum_scaled_jacobian=float("nan")),
    lambda row: row.update(actual_elements=50, element_count_ratio=.005),
))
def test_forged_or_vacuous_evidence_is_rejected(benchmark, tmp_path, change):
    row = record()
    change(row)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark._load_record(path)


def test_nested_alias_does_not_double_count_native_work(benchmark):
    report = work()["faces"]["1"]
    diagnostics = {"native_v2": report, "native_diagnostics": {"native_v2": report}}
    actual = benchmark._native_work(
        {"faces": {"1": diagnostics}}, "frontal", "cylinder_patch")
    assert actual == work()


def test_mapped_zero_use_is_exempt_but_cannot_report_refinement(benchmark):
    empty = {"faces": {}, "insertions": 0, "published_insertions": 0,
             "topology_operations": 0}
    benchmark._require_refinement_work(empty, "frontal", "mapped_zero_use")
    with pytest.raises(ValueError):
        benchmark._require_refinement_work(work(), "frontal", "mapped_zero_use")


def test_wrong_comparison_order_is_rejected(benchmark, tmp_path):
    path = tmp_path / "frontal.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy followed by frontal"):
        benchmark._compare(SimpleNamespace(legacy=path, frontal=path))


def test_sample_journal_preserves_failure_and_never_replays(benchmark, tmp_path):
    args = SimpleNamespace(output=tmp_path / "result.json", case="planar",
                           route="legacy", source_commit="a" * 40)
    core = SimpleNamespace(node_coordinates=np.zeros((3, 3)),
                           triangle_connectivity=np.asarray(((0, 1, 2),)),
                           quad_connectivity=np.empty((0, 4), dtype=int),
                           num_triangles=1, num_quads=0)
    calls = []
    def generate():
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("first failed candidate")
        return core, {}
    with pytest.raises(RuntimeError, match="first failed candidate"):
        benchmark._measure(generate, args)
    assert len(calls) == 3 and not args.output.exists()
    journal = args.output.with_name(args.output.name + ".partial.jsonl")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "warmup", "measurement", "failure"]
    original = journal.read_bytes()
    with pytest.raises(FileExistsError):
        benchmark._measure(generate, args)
    assert len(calls) == 3 and journal.read_bytes() == original


@pytest.mark.parametrize("name", (
    "cylinder_patch", "cylinder_sectors", "cylinder_seam_trim",
))
def test_cylinder_recipe_public_routes_and_shared_identity(benchmark, name):
    from anygeometry import to_dict
    from anymesher.hybrid import generate_hybrid_mesh_result, _neutral_shell_core
    from anymesher.quality_v2 import assert_valid_mesh

    fixture = benchmark.cylinder_case(name)
    before = benchmark._digest_contract(to_dict(fixture.model))
    contracts = []
    for route in ("legacy", "frontal"):
        options = fixture.options(.4, route, 128)
        result = generate_hybrid_mesh_result(
            fixture.model, face_ids=fixture.face_ids, target_size=.4,
            strategy="native", overrides=fixture.overrides(.4),
            refinements=fixture.refinements(.4),
            native_backend="python", native_options=options, recombine=True,
        )
        assert set(result.strategy_by_face.values()) == {"native"}
        assert_valid_mesh(_neutral_shell_core(result.mesh))
        contracts.append(benchmark._digest_contract(fixture.mesh_contract(result.mesh, .4)))
        assert benchmark._digest_contract(to_dict(fixture.model)) == before
        receipt = benchmark._native_work(
            {"faces": {str(face): dict(row) for face, row in
                       result.triangulation_backend_by_face.items()}}, route, name)
        assert bool(receipt["insertions"]) is (route == "frontal")
        broken = copy(result.mesh)
        broken.nodes_of_edge = dict(result.mesh.nodes_of_edge)
        edge = next(iter(broken.nodes_of_edge))
        broken.nodes_of_edge[edge] = broken.nodes_of_edge[edge][:-1]
        with pytest.raises(ValueError, match="protected station"):
            fixture.mesh_contract(broken, .4)
    assert contracts[0] == contracts[1]
    from anymesher import MetricFieldSpec
    zones = fixture.refinements(.4)
    size_field = SimpleNamespace(target_size=.4, zones=zones,
                                 _sources=tuple(zone.sources(fixture.model) for zone in zones))
    assert fixture.options(.4, "legacy", 128).metric_field is None
    assert MetricFieldSpec.from_size_field(size_field).to_dict() == (
        fixture.metric_spec(.4).to_dict())
    assert fixture.options(.4, "frontal", 128).metric_field is None


@pytest.mark.parametrize("mutation", (
    "discarded", "missing_publication", "overreported", "cancelled", "missing_completion",
))
def test_attempted_work_cannot_stand_in_for_publication(benchmark, mutation):
    row = work()
    face = row["faces"]["1"]
    if mutation == "discarded":
        face["published_insertions"] = row["published_insertions"] = 0
    elif mutation == "missing_publication":
        face.pop("published_insertions")
    elif mutation == "overreported":
        face["published_insertions"] = row["published_insertions"] = 4
    elif mutation == "cancelled":
        face["cancelled"] = True
    else:
        face.pop("cancelled")
    with pytest.raises(ValueError):
        benchmark._require_refinement_work(row, "frontal", "cylinder_patch")
