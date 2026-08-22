from __future__ import annotations

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.surface_mesh import (
    SurfaceMeshOptions,
    _target_points,
    mesh_planar_surface,
)


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(values, dtype=float), 12)
    order = np.lexsort((rounded[:, 1], rounded[:, 0]))
    return rounded[order]


def test_public_native_quality_defaults_remain_0_2_3_compatible() -> None:
    settings = SurfaceMeshOptions()

    assert settings.min_scaled_jacobian == 0.20
    assert settings.max_aspect_ratio == 4.0
    assert settings.min_angle == 30.0
    assert settings.max_angle == 150.0
    assert settings.max_warpage == 0.10


@pytest.mark.parametrize(
    "tensor,match",
    (
        (((1.0, 0.5), (0.0, 1.0)), "symmetric positive definite"),
        (((1.0, 0.0), (0.0, 0.0)), "symmetric positive definite"),
        (((1.0, 0.0), (0.0, 1000.0)), "aspect ratio"),
    ),
)
def test_metric_tensor_is_strictly_spd_and_bounded(
    tensor: tuple[tuple[float, float], tuple[float, float]], match: str
) -> None:
    with pytest.raises(MeshError, match=match):
        SurfaceMeshOptions(metric_tensor=tensor)


def test_metric_target_lattice_is_rotation_covariant() -> None:
    outer = np.asarray(((-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)))
    metric = np.diag((16.0, 4.0))
    first = _target_points(outer, (), (), 0.25, metric_tensor=metric)
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)))
    rotated = _target_points(
        outer @ rotation.T,
        (),
        (),
        0.25,
        metric_tensor=rotation @ metric @ rotation.T,
    )

    np.testing.assert_allclose(
        _sorted_rows(first),
        _sorted_rows(rotated @ rotation),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_target_lattice_budget_blocks_work_before_allocation() -> None:
    outer = np.asarray(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))

    with pytest.raises(MeshError, match="above max_lattice_points"):
        _target_points(
            outer,
            (),
            (),
            0.001,
            max_lattice_points=1000,
        )


def test_target_lattice_checks_cancellation_during_generation() -> None:
    phases: list[str] = []

    def cancel(phase: str) -> None:
        phases.append(phase)
        if phase == "native target lattice":
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        _target_points(
            np.asarray(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))),
            (),
            (),
            0.1,
            cancellation_check=cancel,
        )

    assert phases == ["native target lattice"]


def test_opt_in_quality_policy_fails_closed_with_actual_element_ids() -> None:
    acute = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.02, 0.10)))

    with pytest.raises(MeshError, match=r"minimum_angle=1.*poor element IDs"):
        mesh_planar_surface(
            acute,
            options=SurfaceMeshOptions(
                target_size=0.2,
                backend="python",
                recombine=False,
                enforce_quality=True,
                min_angle=30.0,
            ),
        )


def test_quality_and_recombination_diagnostics_use_selected_policy() -> None:
    diagnostics: dict[str, object] = {}
    mesh_planar_surface(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        options=SurfaceMeshOptions(
            target_size=0.25,
            backend="python",
            min_angle=25.0,
        ),
        diagnostics=diagnostics,
    )

    quality = diagnostics["quality_policy"]
    recombination = diagnostics["recombination"]
    assert isinstance(quality, dict)
    assert quality["thresholds"]["minimum_angle"] == 25.0
    assert isinstance(quality["violation_counts"]["minimum_angle"], int)
    assert isinstance(quality["poor_element_ids"], list)
    assert isinstance(recombination, dict)
    assert recombination["exchange_work"] <= 1_000_000
