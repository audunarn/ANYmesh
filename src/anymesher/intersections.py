"""Qualified ANYgeometry intersection planning and legacy migration seam.

Production meshing is read-only and consumes persistent Sheets, Members,
Attachments and Junctions. Geometry mutation is available here only through
ANYgeometry's public query/plan/apply protocol with an explicit
``ConnectionIntent``. Typed unsupported or capability-missing results fail
closed in the kernel; ANYmesher never treats an empty payload as disjointness.
"""

from __future__ import annotations

import warnings
from typing import Any

from anygeometry import (
    ConnectionIntent,
    GeometryModel,
    ImprintApplication,
    ImprintPlan,
    apply_imprint,
    plan_imprint,
    query_intersection,
)

__all__ = [
    "apply_intersection_mutation",
    "generate_mesh_with_intersections",
    "generate_mesh_with_legacy_imprint",
    "plan_intersection_mutation",
]


def _intent(value: ConnectionIntent | str) -> ConnectionIntent:
    if isinstance(value, ConnectionIntent):
        return value
    try:
        return ConnectionIntent(str(value).strip().lower())
    except ValueError as error:
        expected = ", ".join(item.value for item in ConnectionIntent)
        raise ValueError(
            f"unknown connection intent {value!r}; expected one of {expected}"
        ) from error


def plan_intersection_mutation(
    geometry: GeometryModel,
    first: Any,
    second: Any,
    *,
    intent: ConnectionIntent | str,
) -> ImprintPlan:
    """Return the kernel-qualified, revision-bound mutation plan."""

    policy = _intent(intent)
    result = query_intersection(geometry, first, second)
    return plan_imprint(geometry, result, policy=policy)


def apply_intersection_mutation(
    geometry: GeometryModel,
    first: Any,
    second: Any,
    *,
    intent: ConnectionIntent | str,
) -> ImprintApplication:
    """Apply one explicit kernel plan and return its authoritative ChangeSet."""

    policy = _intent(intent)
    plan = plan_intersection_mutation(geometry, first, second, intent=policy)
    return apply_imprint(geometry, plan, policy=policy)


def generate_mesh_with_legacy_imprint(
    geometry: GeometryModel, **options: Any
):
    """Run the quarantined pre-0.2 clone/imprint meshing implementation."""

    from ._legacy_intersections import generate_mesh_with_intersections

    return generate_mesh_with_intersections(geometry, **options)


def generate_mesh_with_intersections(
    geometry: GeometryModel, **options: Any
):
    """Deprecated alias for :func:`generate_mesh_with_legacy_imprint`."""

    warnings.warn(
        "generate_mesh_with_intersections uses the quarantined legacy "
        "clone/imprint workflow; use generate_hybrid_mesh with persistent "
        "ANYgeometry structural records, or explicitly call "
        "generate_mesh_with_legacy_imprint during migration",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_mesh_with_legacy_imprint(geometry, **options)
