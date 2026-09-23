"""Read-only consumption of ANYgeometry's qualified sector occurrence atlas.

This boundary does not activate meshing, certify geometry, assign mesh nodes,
unwrap sampled trims, or infer aliases from coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from anygeometry import (
    CylinderAtlasError,
    CylinderAtlasErrorCode,
    CylinderAtlasPolicy,
    CylinderAtlasResult,
    CylinderAtlasStatus,
    EntityHandle,
    evaluate_cylinder_occurrences,
    query_cylinder_atlas,
    validate_cylinder_atlas_binding,
)

from ._cylindrical_chart import CylindricalMetricChart
from .errors import MeshError


def _owner(geometry):
    guard = getattr(geometry, "assert_current", None)
    if callable(guard):
        guard()
        return geometry.source
    return geometry


@dataclass(frozen=True, slots=True)
class CylindricalAtlasBinding:
    """Caller-bound owner evidence and physical charts for separate Faces.

    Source geometry remains externally owned. Every evaluation delegates fresh
    binding checks to the owner. Chart consumers must validate this binding at
    their transaction boundary; charts independently reject stale revisions.
    FaceUse orientation and external radial incidence remain in ``atlas``.
    """

    geometry: object
    face_uses: tuple[EntityHandle, ...]
    reference_face_use: EntityHandle
    atlas: CylinderAtlasResult
    charts: tuple[tuple[EntityHandle, CylindricalMetricChart], ...]

    @property
    def model_id(self):
        return self.atlas.model_id

    @property
    def revision(self):
        return self.atlas.revision

    @property
    def face_records(self):
        return self.atlas.sectors

    @property
    def certification_kind(self):
        return "full_period_sector_atlas"

    def validate(self, *, cancellation_check: Callable[[str], None] | None = None):
        validate_cylinder_atlas_binding(
            _owner(self.geometry),
            self.atlas,
            self.face_uses,
            reference_face_use=self.reference_face_use,
            expected_revision=self.atlas.revision,
            cancellation_check=cancellation_check,
        )

    def evaluate(self, requests, *, cancellation_check=None):
        """Return owner rows unchanged, including identity keys and enclosures."""
        return evaluate_cylinder_occurrences(
            _owner(self.geometry),
            self.atlas,
            requests,
            face_uses=self.face_uses,
            reference_face_use=self.reference_face_use,
            expected_revision=self.atlas.revision,
            cancellation_check=cancellation_check,
        )

    def chart_for(self, face_use: EntityHandle, *, cancellation_check=None):
        self.validate(cancellation_check=cancellation_check)
        for selected, chart in self.charts:
            if selected == face_use:
                chart._current()
                return chart
        raise MeshError("face use is not selected in this cylindrical atlas")


def prepare_cylindrical_atlas(
    geometry,
    face_uses,
    *,
    reference_face_use: EntityHandle,
    policy: CylinderAtlasPolicy | None = None,
    cancellation_check: Callable[[str], None] | None = None,
) -> CylindricalAtlasBinding:
    """Prepare physical sector charts only after owner qualification succeeds.

    The owner bounds selection consumption and decides all unsupported,
    unresolved and invalid cases. Its typed and operational failures propagate
    unchanged; no partially constructed binding is returned.
    """
    owner = _owner(geometry)
    atlas = query_cylinder_atlas(
        owner,
        face_uses,
        reference_face_use=reference_face_use,
        expected_revision=owner.revision,
        policy=policy,
        cancellation_check=cancellation_check,
    )
    selected = atlas.requested_face_uses
    validate_cylinder_atlas_binding(
        owner,
        atlas,
        selected,
        reference_face_use=reference_face_use,
        expected_revision=atlas.revision,
        cancellation_check=cancellation_check,
    )
    if atlas.status is not CylinderAtlasStatus.QUALIFIED or not atlas.certificate.complete:
        raise CylinderAtlasError(
            CylinderAtlasErrorCode.UNQUALIFIED_RESULT,
            tuple(atlas.diagnostics) or ("cylindrical atlas is not qualified and complete",),
        )
    charts = []
    for sector in atlas.sectors:
        if cancellation_check is not None:
            cancellation_check("cylindrical atlas physical sector chart")
        charts.append(
            (sector.face_use, CylindricalMetricChart.from_geometry(geometry, sector.face.id))
        )
    binding = CylindricalAtlasBinding(
        geometry, selected, reference_face_use, atlas, tuple(charts)
    )
    if cancellation_check is not None:
        cancellation_check("cylindrical atlas preparation complete")
    # The last caller callback may mutate the owner or leave a transaction open.
    # Revalidate authoritatively without invoking caller code again afterward.
    binding.validate()
    return binding
