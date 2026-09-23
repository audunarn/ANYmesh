"""Registered, work-bearing native-v2 development fixtures.

Fixture preparation is deterministic. No geometry is built until a caller has
passed the development scope guard; no benchmark here selects a large workload.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


CASE_FAMILIES = {
    "planar_callback_present": ("planar", True),
    "planar_callback_absent": ("planar", False),
    "cylindrical_callback_present": ("cylindrical", True),
    "cylindrical_callback_absent": ("cylindrical", False),
    "mapped_zero_use": ("mapped", False),
    "structural_three_plate": ("structural", False),
    "plate_with_hole": ("hole", False),
    "planar_insertion_work": ("planar_insertion", False),
}


@dataclass(frozen=True)
class PreparedCase:
    name: str
    family: str
    geometry: Any
    target_size: float
    native_options: Any
    refinements: tuple[Any, ...]
    recombine: bool
    strategy: str
    callback_present: bool


def prepare_case(name: str, requested_elements: int, route: str) -> PreparedCase:
    if name not in CASE_FAMILIES:
        raise ValueError("unregistered development case")
    if route not in {"legacy", "frontal"}:
        raise ValueError("unregistered development route")
    if requested_elements not in {1_000, 10_000, 100_000}:
        raise ValueError("unregistered development element request")

    from anymesher import (
        FeatureDistanceMetricControl, IsotropicMetricControl,
        MetricFieldSpec, NativeMeshingOptions,
    )
    from anymesher.refinement import Refinement
    from native_hybrid_performance import (
        _pentagon, _rectangle, _three_plate_intersection, _plate_with_hole,
    )
    from native_v2_cylinder_cases import cylinder_case

    family, callback_present = CASE_FAMILIES[name]
    if family == "cylindrical":
        fixture = cylinder_case("cylinder_patch")
        geometry, area = fixture.model, fixture.area
        center = fixture.centers[0]
        factor = 1.4
    elif family in {"planar", "planar_insertion"}:
        geometry, area, center, factor = _pentagon(), 2.377641290737884, (0., 0., 0.), 1.65
    elif family == "mapped":
        geometry, area, center, factor = _rectangle(), 1., (0.5, 0.5, 0.), 1.
    elif family == "structural":
        geometry = _three_plate_intersection()
        area, center, factor = 8. + 2. * math.sqrt(2.), (1., 1., 0.), 1.65
    else:
        geometry = _plate_with_hole()
        area, center, factor = 16. - math.pi * .25, (2.75, 2., 0.), 1.65
    target = factor * math.sqrt(area / requested_elements)
    refinements = () if family in {"mapped", "planar_insertion"} else (
        Refinement(size=target * .5, radius=target * 2., center=center,
                   growth=1.5, name="development-work"),
    )
    if family == "cylindrical":
        options = fixture.options(target, route, requested_elements)
    elif family == "planar_insertion":
        if route != "frontal":
            raise ValueError("insertion-work fixture requires frontal placement")
        metric = MetricFieldSpec(
            IsotropicMetricControl(target),
            feature_controls=(FeatureDistanceMetricControl(
                ((0., 0., 0.),), target * .7, target * 2., 1.5,
                "insertion-work",
            ),),
        )
        options = NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial", metric_field=metric,
            max_insertions=max(128, requested_elements // 2),
        )
    else:
        options = (
            NativeMeshingOptions() if route == "legacy" else
            NativeMeshingOptions(point_placement="frontal_delaunay",
                                 metric_mode="isotropic_spatial",
                                 max_insertions=max(128, requested_elements // 2))
        )
    return PreparedCase(
        name, family, geometry, target, options, refinements,
        family == "hole", "mapped" if family == "mapped" else "native",
        callback_present,
    )


def generate_once(
    prepared: PreparedCase, backend: str,
    *, cancellation_stages: list[str] | None = None,
    generator: Callable[..., Any] | None = None,
) -> Any:
    from anymesher.hybrid import generate_hybrid_mesh

    if backend not in {"auto", "python", "compiled"}:
        raise ValueError("unregistered native backend")
    selected_backend = "native" if backend == "compiled" else backend
    stages = cancellation_stages if cancellation_stages is not None else []

    def checkpoint(stage: str) -> None:
        stages.append(stage)

    run = generate_hybrid_mesh if generator is None else generator
    return run(
        prepared.geometry, target_size=prepared.target_size,
        strategy=prepared.strategy, native_backend=selected_backend,
        native_options=prepared.native_options,
        refinements=prepared.refinements, recombine=prepared.recombine,
        cancellation_check=checkpoint if prepared.callback_present else None,
    )
