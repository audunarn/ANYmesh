"""Small authorization-format fixtures; never run benchmark workloads."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "native_t3_runtime_scaling.py"
    spec = importlib.util.spec_from_file_location("runtime_scaling_authority_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timestamp_request_status_ledger_row_is_accepted(benchmark):
    request = "123717860b46472bb527a057dd5f138e"
    text = (
        "| Timestamp | Request ID | Status | Notes |\n"
        "| --- | --- | --- | --- |\n"
        f"| 2026-09-06T08:00:00+02:00 | {request} | APPROVED | bounded run |\n"
    )
    assert benchmark._ledger_approves(text, request)


@pytest.mark.parametrize("status", ["PENDING", "REJECTED", "CHANGES REQUIRED"])
def test_non_approved_status_is_rejected(benchmark, status):
    request = "123717860b46472bb527a057dd5f138e"
    assert not benchmark._ledger_approves(
        f"| 2026-09-06T08:00:00+02:00 | {request} | {status} | APPROVED elsewhere |",
        request,
    )


def test_wrong_request_and_shifted_columns_are_rejected(benchmark):
    request = "123717860b46472bb527a057dd5f138e"
    assert not benchmark._ledger_approves(
        "| 2026-09-06 | ffffffffffffffffffffffffffffffff | APPROVED | bounded |", request
    )
    assert not benchmark._ledger_approves(f"| {request} | APPROVED | bounded |", request)
