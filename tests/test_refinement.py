"""Input and normalization contracts for owner-side refinement zones."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher import EntityRef, Refinement, refine_around, refine_at


def test_valid_refinement_values_are_normalized():
    zone = Refinement(
        size=np.float64(0.1),
        radius=2,
        growth=np.float32(1.4),
        center=np.array([1, 2, 3], dtype=np.int64),
    )

    assert zone.size == pytest.approx(0.1)
    assert zone.radius == 2.0
    assert zone.growth == pytest.approx(1.4)
    assert zone.center == (1.0, 2.0, 3.0)
    assert all(type(value) is float for value in (*zone.center, zone.size, zone.radius, zone.growth))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size", np.nan),
        ("size", True),
        ("size", "0.1"),
        ("radius", np.inf),
        ("radius", False),
        ("radius", 1.0 + 0.0j),
        ("growth", -np.inf),
        ("growth", True),
        ("growth", "1.5"),
    ],
)
def test_refinement_scalars_must_be_finite_real_numbers(field, value):
    values = {"size": 0.1, "radius": 0.0, "growth": 1.5}
    values[field] = value

    with pytest.raises(ValueError, match=rf"{field}.*finite real number"):
        Refinement(**values, center=(0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("size", 0.0, "element size must be positive"),
        ("size", -0.1, "element size must be positive"),
        ("radius", -0.1, "radius must not be negative"),
        ("growth", 1.0, "growth must exceed 1.0"),
        ("growth", 0.5, "growth must exceed 1.0"),
    ],
)
def test_refinement_preserves_scalar_range_rules(field, value, message):
    values = {"size": 0.1, "radius": 0.0, "growth": 1.5}
    values[field] = value

    with pytest.raises(ValueError, match=message):
        Refinement(**values, center=(0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "center",
    [
        (0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, np.nan, 0.0),
        (0.0, np.inf, 0.0),
        (0.0, True, 0.0),
        (0.0, "1.0", 0.0),
    ],
)
def test_refinement_center_must_be_an_exact_finite_xyz_vector(center):
    with pytest.raises(ValueError, match="center needs exactly three finite real"):
        Refinement(size=0.1, center=center)


def test_convenience_builders_do_not_coerce_boolean_inputs():
    ref = EntityRef("vertex", 1)

    with pytest.raises(ValueError, match="element size.*finite real number"):
        refine_around(ref, size=True)
    with pytest.raises(ValueError, match="radius.*finite real number"):
        refine_at((0.0, 0.0, 0.0), size=0.1, radius=False)
    with pytest.raises(ValueError, match="center needs exactly three finite real"):
        refine_at((0.0, True, 0.0), size=0.1)


def test_refinement_still_requires_exactly_one_location_source():
    ref = EntityRef("vertex", 1)

    with pytest.raises(ValueError, match="either ref.*or center"):
        Refinement(size=0.1)
    with pytest.raises(ValueError, match="either ref.*or center"):
        Refinement(size=0.1, ref=ref, center=(0.0, 0.0, 0.0))
