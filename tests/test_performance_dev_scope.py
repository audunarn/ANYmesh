from argparse import Namespace
from copy import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def development_guard(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("performance_dev").guard_request


def request():
    return Namespace(
        case="planar_callback_absent",
        scale="10k", requested_elements=10_000, allow_large=False,
        workstation_elements=None, worker_count=1, checkpoint=False,
    )


def test_default_and_explicit_checkpoint_are_bounded(development_guard):
    assert development_guard(request()) == 10_000
    small = request()
    small.scale, small.requested_elements = "1k", 1_000
    assert development_guard(small) == 1_000
    row = request()
    row.scale, row.requested_elements, row.checkpoint = "100k", 100_000, True
    assert development_guard(row) == 100_000


@pytest.mark.parametrize("change", (
    lambda row: setattr(row, "scale", "500k"),
    lambda row: setattr(row, "scale", "workstation"),
    lambda row: setattr(row, "requested_elements", 500_000),
    lambda row: setattr(row, "requested_elements", 100_001),
    lambda row: setattr(row, "allow_large", True),
    lambda row: setattr(row, "workstation_elements", 500_000),
))
def test_excluded_requests_fail_before_worker_creation(development_guard, change):
    row = copy(request())
    change(row)
    with pytest.raises(ValueError, match="EXCLUDED_BY_SCOPE"):
        development_guard(row)


def test_100k_is_not_an_implicit_iteration_request(development_guard):
    row = request()
    row.scale, row.requested_elements = "100k", 100_000
    with pytest.raises(ValueError, match="explicit checkpoint"):
        development_guard(row)


def test_wheel_identity_is_required_and_source_identity_is_exclusive(development_guard):
    row = request()
    row.install_kind, row.wheel_sha256 = "wheel", None
    with pytest.raises(ValueError, match="require a wheel SHA-256"):
        development_guard(row)
    row.wheel_sha256 = "a" * 64
    assert development_guard(row) == 10_000
    row.install_kind = "source"
    with pytest.raises(ValueError, match="must not carry"):
        development_guard(row)


@pytest.mark.parametrize("case", ("renamed_500k", "workstation_planar", "unknown"))
def test_unregistered_case_names_fail_before_worker_creation(development_guard, case):
    row = request()
    row.case = case
    with pytest.raises(ValueError, match="EXCLUDED_BY_SCOPE"):
        development_guard(row)
