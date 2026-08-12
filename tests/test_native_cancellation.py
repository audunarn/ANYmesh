"""Safe-phase cancellation for native hybrid generation."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import GeometryModel, Plane
from anymesher.hybrid import generate_hybrid_mesh


class _Cancelled(RuntimeError):
    pass


def test_native_generation_checks_cancellation_after_real_triangulation() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 0.4, 0.0), (0.0, 1.0, 0.0))
    )
    geometry.add_face(
        geometry.add_polyline(vertices, close=True),
        surface=Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    stages: list[str] = []

    def cancel_after_triangulation(stage: str) -> None:
        stages.append(stage)
        if stage == "native surface triangulation complete":
            raise _Cancelled(stage)

    with pytest.raises(_Cancelled, match="triangulation complete"):
        generate_hybrid_mesh(
            geometry,
            target_size=0.2,
            strategy="native",
            cancellation_check=cancel_after_triangulation,
        )

    assert stages[-1] == "native surface triangulation complete"
    assert "native surface recombination complete" not in stages
