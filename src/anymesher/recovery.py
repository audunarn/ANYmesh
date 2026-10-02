"""Bounded, solver-aware recovery over the public hybrid mesh engines.

This opt-in controller does not change the meaning of explicit hybrid calls.
Each recipe owns a detached geometry copy, so a rejected candidate cannot
change the source or the next attempt's boundary identities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from time import monotonic
from typing import Any, Mapping

from anygeometry import GeometryModel, clone_prepared_geometry
try:
    from anygeometry.surfaces import ExtrudedSurface
except ImportError:  # Older supported ANYgeometry builds predate schema 6.
    ExtrudedSurface = ()

from .errors import MeshError, StructuredQualityRejected
from .quad.validate import QuadQualityRejected
from .hybrid import HybridMeshResult, generate_hybrid_mesh_result
from .quality import verify_mesh_quality
from .quad.front import FrontNoCandidate, FrontRejected
from .quad.public_integration import QuadCapabilityMissing, QuadPublicUnsupported
from .s3_quality import S3QualityError
from .s3_repair import S3RepairError


AUTOMATION_CONTRACT_ID = "ANYMESHER_AUTOMATIC_RECOVERY_V1"


class MeshRecoveryIncomplete(MeshError):
    """The shared operation budget expired before a candidate was accepted."""


@dataclass(frozen=True, slots=True)
class MeshAutomationOptions:
    """An explicit choice between strict and recoverable method selection."""

    strict_method: bool = False
    allow_inspection: bool = True
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        if type(self.strict_method) is not bool or type(self.allow_inspection) is not bool:
            raise TypeError("automation switches must be Boolean")
        if self.max_seconds is not None:
            if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)):
                raise TypeError("max_seconds must be numeric")
            if not isfinite(float(self.max_seconds)) or float(self.max_seconds) <= 0.0:
                raise ValueError("max_seconds must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_id": AUTOMATION_CONTRACT_ID,
            "strict_method": self.strict_method,
            "allow_inspection": self.allow_inspection,
            "max_seconds": self.max_seconds,
        }


@dataclass(frozen=True, slots=True)
class AutomaticMeshResult:
    result: HybridMeshResult
    selected_method: str
    status: str
    attempts: tuple[Mapping[str, object], ...]

    @property
    def mesh(self):
        return self.result.mesh

    def to_dict(self) -> dict[str, object]:
        quality = verify_mesh_quality(self.mesh)
        return {
            "contract_id": AUTOMATION_CONTRACT_ID,
            "status": self.status,
            "mesh_validity": "VALID",
            "selected_method": self.selected_method,
            "element_families": {
                "quads": len(self.mesh.quads),
                "triangles": len(self.mesh.tris),
                "beams": len(self.mesh.beams),
                "couplings": len(self.mesh.couplings),
            },
            "quality_warnings": list(quality.warnings),
            "attempts": [dict(attempt) for attempt in self.attempts],
            "solver_admission": "ADMITTED" if self.status == "ready" else "BLOCKED",
        }


_RECOVERABLE = (
    S3QualityError,
    QuadQualityRejected,
    StructuredQualityRejected,
    FrontNoCandidate,
    FrontRejected,
    QuadPublicUnsupported,
    QuadCapabilityMissing,
)


def _route_name(options: Mapping[str, Any]) -> str:
    if options.get("quad_options") is not None:
        return "quad_first"
    return str(getattr(options.get("strategy", "auto"), "value", options.get("strategy", "auto")))


def _missing_extruded_quadratic_certificate(
    geometry: GeometryModel, options: Mapping[str, Any], result: HybridMeshResult,
) -> bool:
    """Do not admit a new exact owner through an uncertified legacy fallback."""
    if options.get("order", "linear") != "quadratic":
        return False
    selected = options.get("face_ids")
    face_ids = geometry.faces if selected is None else selected
    if not any(
        face_id in geometry.faces
        and isinstance(geometry.faces[face_id].surface, ExtrudedSurface)
        for face_id in face_ids
    ):
        return False
    certificate = result.mesh.hybrid_diagnostics.get("high_order_geometry")
    return not (
        isinstance(certificate, Mapping)
        and certificate.get("status") == "CERTIFIED_POSITIVE"
    )


def _attempt_options(
    first: Mapping[str, Any], fallback_structured_options: Any | None
) -> tuple[tuple[str, dict[str, Any]], ...]:
    preferred = _route_name(first)
    base = dict(first)
    routes: list[tuple[str, dict[str, Any]]] = [(preferred, base)]
    if preferred != "auto":
        automatic = dict(base)
        automatic.update(
            strategy="auto", quad_options=None, layout_policy="existing",
            structured_options=fallback_structured_options,
        )
        routes.append(("auto", automatic))
    if preferred != "native" or base.get("quad_options") is not None:
        native = dict(base)
        native.update(
            strategy="native", quad_options=None, layout_policy="existing",
            structured_options=None, recombine=False,
        )
        if all(route != "native" or options.get("quad_options") is not None for route, options in routes):
            routes.append(("native", native))
    return tuple(routes[:3])


def _prefer_analytic_trim_recipe(geometry, first, recipes):
    """Select an existing native recipe for automatic analytic material cuts.

    This selects discretization only. Hybrid generation must still obtain the
    owner's revision-bound material charts and pass unchanged admission gates.
    Explicit native/mapped requests, spatial controls and strict requests retain
    their route. Historical circular cylinder patches retain their route too.
    """
    if (_route_name(first) not in {'auto','quad_first'}
            or first.get('order','linear')!='linear'):
        return recipes
    import anygeometry as owner
    types=tuple(kind for name in ('EllipticArc','CylinderIntersectionCurve')
                if isinstance((kind:=getattr(owner,name,None)),type))
    if not types or not any(isinstance(edge.curve,types) for edge in geometry.edges.values()):
        return recipes
    from .native_v2 import NativeMeshingOptions
    options=NativeMeshingOptions.coerce(first.get('native_options'))
    if options.point_placement!='legacy_lattice':
        return recipes
    native=next((dict(recipe) for method,recipe in recipes if method=='native'),None)
    if native is None:
        return recipes
    # The automatic analytic route must carry the existing automatic quality
    # policy rather than silently adopting the direct surface API's preference.
    from .structured import StructuredMeshingOptions
    native.setdefault('_native_surface_options',
                      StructuredMeshingOptions.create(first.get('structured_options')))
    native['native_options']=replace(options,point_placement='frontal_delaunay',
                                     metric_mode='isotropic_spatial')
    return (('native',native),*(item for item in recipes if item[0]!='native'))


def generate_automatic_mesh_result(
    geometry: GeometryModel,
    *,
    automation: MeshAutomationOptions | None = None,
    fallback_structured_options: Any | None = None,
    **meshing_options: Any,
) -> AutomaticMeshResult:
    """Try bounded methods and retain a valid inspect-only mesh if needed.

    Solver admission is requested through the established ``qualified_s3``
    switch. No weaker profile is inferred or selected by this function.
    """

    policy = MeshAutomationOptions() if automation is None else automation
    if not isinstance(policy, MeshAutomationOptions):
        raise TypeError("automation must be MeshAutomationOptions")
    deadline = (
        None if policy.max_seconds is None else monotonic() + policy.max_seconds
    )

    def check_budget(phase: str) -> None:
        if deadline is not None and monotonic() >= deadline:
            raise MeshRecoveryIncomplete(
                f"automatic meshing time budget expired during {phase}"
            )
        callback = meshing_options.get("cancellation_check")
        if callback is not None:
            callback(phase)

    check_budget("source geometry validation")
    problems = geometry.validate_topology()
    if problems:
        raise MeshError("invalid source geometry: " + "; ".join(problems))
    first = dict(meshing_options)
    recipes = _attempt_options(first, fallback_structured_options)
    if policy.strict_method:
        recipes = recipes[:1]
    else:
        recipes = _prefer_analytic_trim_recipe(geometry, first, recipes)
    attempts: list[Mapping[str, object]] = []
    first_recoverable: Exception | None = None
    admission_error: S3QualityError | None = None
    inspectable_result: HybridMeshResult | None = None
    inspectable_method: str | None = None
    inspection_reason: str | None = None
    for method, recipe in recipes:
        check_budget(f"automatic meshing: {method}")
        recipe["cancellation_check"] = check_budget
        try:
            result = generate_hybrid_mesh_result(clone_prepared_geometry(geometry), **recipe)
        except MeshError as error:
            mapped_rejection = (
                method == "mapped"
                and str(error).startswith("explicit mapped strategy")
            )
            if policy.strict_method or (
                not isinstance(error, _RECOVERABLE) and not mapped_rejection
            ):
                raise
            if first_recoverable is None:
                first_recoverable = error
            if isinstance(error, S3QualityError) and admission_error is None:
                admission_error = error
            candidate = getattr(error, "inspectable_result", None)
            if inspectable_result is None and isinstance(candidate, HybridMeshResult):
                inspectable_result = candidate
                inspectable_method = method
            admission = getattr(error, "admission", None)
            failing_ids = (
                [] if admission is None else [
                    int(item.element_id) for item in admission.elements
                    if not item.admitted
                ]
            )
            attempts.append({
                "method": method,
                "status": "rejected",
                "error_type": type(error).__name__,
                "reason": str(error),
                "problem_element_ids": failing_ids,
            })
            continue
        if method != "quad_first" and _missing_extruded_quadratic_certificate(
            geometry, first, result
        ):
            reason = (
                "quadratic exact-extrusion fallback lacks a CERTIFIED_POSITIVE "
                "high-order geometry certificate"
            )
            attempts.append({
                "method": method,
                "status": "rejected",
                "error_type": "MissingHighOrderCertificate",
                "reason": reason,
                "problem_element_ids": [],
            })
            if inspectable_result is None:
                inspectable_result = result
                inspectable_method = method
                inspection_reason = reason
            if first_recoverable is None:
                first_recoverable = MeshError(reason)
            continue
        attempts.append({"method": method, "status": "selected"})
        selected = AutomaticMeshResult(result, method, "ready", tuple(attempts))
        result.mesh.hybrid_diagnostics["automation"] = selected.to_dict()
        return selected

    if policy.allow_inspection and (
        admission_error is not None or inspection_reason is not None
    ):
        # An admission failure happens after mesh construction. Regenerate the
        # preferred recipe without solver admission, then publish it only as an
        # inspection artifact. This path is deliberately last, not a fallback
        # solver route.
        method, recipe = recipes[0]
        candidate = inspectable_result
        if candidate is not None and inspectable_method is not None:
            method = inspectable_method
        if candidate is None:
            candidate_options = dict(recipe)
            candidate_options["qualified_s3"] = False
            check_budget("inspection candidate")
            try:
                candidate = generate_hybrid_mesh_result(
                    clone_prepared_geometry(geometry), **candidate_options
                )
            except _RECOVERABLE:
                candidate = None
        if candidate is not None:
            quality = verify_mesh_quality(candidate.mesh)
            attempts.append({
                "method": method,
                "status": "inspection_only",
                "problem_element_ids": sorted(set(quality.poor_element_ids) | set(
                    attempts[0].get("problem_element_ids", ())
                )),
                "quality_warnings": list(quality.warnings),
                "reason": (
                    str(admission_error) if admission_error is not None
                    else inspection_reason
                ),
            })
            selected = AutomaticMeshResult(
                candidate, method, "inspection_only", tuple(attempts)
            )
            candidate.mesh.hybrid_diagnostics["automation"] = selected.to_dict()
            return selected
    if first_recoverable is not None:
        raise first_recoverable
    raise MeshError("automatic meshing exhausted all supported routes")
