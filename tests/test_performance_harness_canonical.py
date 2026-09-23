"""Checkout newline representation must not change benchmark provenance."""

import importlib
from pathlib import Path


def test_harness_identity_ignores_crlf_but_detects_code_change(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    worker = importlib.import_module("performance_dev_measure")
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    names = (
        "native_hybrid_performance.py",
        "native_v2_cylinder_cases.py",
        "performance_dev.py",
        "performance_dev_cases.py",
        "performance_dev_measure.py",
    )
    for name in names:
        (bench / name).write_bytes(b"value = 1\n")
    expected = worker._harness_digest(tmp_path)
    for name in names:
        (bench / name).write_bytes(b"value = 1\r\n")
    assert worker._harness_digest(tmp_path) == expected
    (bench / names[0]).write_bytes(b"value = 2\r\n")
    assert worker._harness_digest(tmp_path) != expected
