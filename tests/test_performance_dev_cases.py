"""The registered small fixtures keep callback transport and size pairing exact."""

from pathlib import Path
import importlib

import pytest


@pytest.fixture
def cases(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("performance_dev_cases")


def test_planar_callback_lanes_share_geometry_sizing_and_options(cases):
    present = cases.prepare_case("planar_callback_present", 1_000, "frontal")
    absent = cases.prepare_case("planar_callback_absent", 1_000, "frontal")
    assert present.target_size == absent.target_size
    assert present.native_options == absent.native_options
    assert present.strategy == absent.strategy == "native"
    assert present.callback_present and not absent.callback_present


def test_cylinder_uses_physical_target_factor(cases):
    cylinder = cases.prepare_case("cylindrical_callback_absent", 1_000, "frontal")
    assert cylinder.target_size > 0
    assert cylinder.target_size == pytest.approx(1.4 * (cylinder.geometry.faces[next(iter(cylinder.geometry.faces))].surface.radius * 3.141592653589793 / 2 / 1_000) ** .5)
    assert cylinder.strategy == "native"


@pytest.mark.parametrize("name", ("mapped_zero_use", "structural_three_plate", "plate_with_hole"))
def test_real_fixture_preparation_is_bounded(cases, name):
    prepared = cases.prepare_case(name, 1_000, "legacy")
    assert prepared.target_size > 0
    assert prepared.geometry.faces
    assert prepared.native_options.point_placement == "legacy_lattice"


def test_callback_is_present_only_in_the_named_lane(cases):
    observed = []

    def fake(_geometry, **kwargs):
        observed.append(kwargs)
        callback = kwargs["cancellation_check"]
        if callback is not None:
            callback("fixture checkpoint")
        return object()

    stages = []
    for name in ("planar_callback_present", "planar_callback_absent"):
        fixture = cases.prepare_case(name, 1_000, "legacy")
        cases.generate_once(fixture, "python", cancellation_stages=stages, generator=fake)
    assert stages == ["fixture checkpoint"]
    assert callable(observed[0]["cancellation_check"])
    assert observed[1]["cancellation_check"] is None


def test_insertion_fixture_keeps_seed_and_metric_sizing_distinct(cases):
    prepared = cases.prepare_case("planar_insertion_work", 1_000, "frontal")
    assert not prepared.refinements
    assert prepared.native_options.metric_field.feature_controls[0].target_size < prepared.target_size
    with pytest.raises(ValueError, match="requires frontal"):
        cases.prepare_case("planar_insertion_work", 1_000, "legacy")
