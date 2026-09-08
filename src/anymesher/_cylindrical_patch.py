"""Owner-certified nonperiodic cylinder patches, not periodic atlas substitutes."""

from dataclasses import dataclass

from anygeometry import (
    CylinderPatchError,
    CylinderPatchErrorCode,
    CylinderPatchResult,
    CylinderPatchStatus,
    EntityHandle,
    evaluate_cylinder_patch_occurrences,
    query_cylinder_patch,
    validate_cylinder_patch_binding,
)

from ._cylindrical_atlas import _owner
from ._cylindrical_chart import CylindricalMetricChart
from .errors import MeshError


@dataclass(frozen=True, slots=True)
class CylindricalPatchBinding:
    geometry: object
    face_uses: tuple[EntityHandle, ...]
    patch: CylinderPatchResult
    charts: tuple[tuple[EntityHandle, CylindricalMetricChart], ...]

    @property
    def model_id(self):
        return self.patch.model_id

    @property
    def revision(self):
        return self.patch.revision

    @property
    def face_records(self):
        return (self.patch,)

    @property
    def certification_kind(self):
        return "nonperiodic_owner_patch"

    def validate(self, *, cancellation_check=None):
        validate_cylinder_patch_binding(
            _owner(self.geometry), self.patch, self.face_uses,
            expected_revision=self.patch.revision,
            cancellation_check=cancellation_check,
        )

    def evaluate(self, requests, *, cancellation_check=None):
        return evaluate_cylinder_patch_occurrences(
            _owner(self.geometry), self.patch, requests,
            face_uses=self.face_uses, expected_revision=self.patch.revision,
            cancellation_check=cancellation_check,
        )

    def chart_for(self, face_use, *, cancellation_check=None):
        self.validate(cancellation_check=cancellation_check)
        for selected, chart in self.charts:
            if selected == face_use:
                chart._current()
                return chart
        raise MeshError("face use is not selected in this cylindrical patch")


def prepare_cylindrical_patch(geometry, face_uses, *, policy=None, cancellation_check=None):
    """Require a fresh owner proof before making a physical patch chart."""
    owner = _owner(geometry)
    patch = query_cylinder_patch(
        owner, face_uses, expected_revision=owner.revision,
        policy=policy, cancellation_check=cancellation_check,
    )
    selected = patch.requested_face_uses
    validate_cylinder_patch_binding(
        owner, patch, selected, expected_revision=patch.revision,
        cancellation_check=cancellation_check,
    )
    if patch.status is not CylinderPatchStatus.QUALIFIED or not patch.certificate.complete:
        raise CylinderPatchError(
            CylinderPatchErrorCode.UNQUALIFIED_RESULT,
            tuple(patch.diagnostics) or ("cylindrical patch is not qualified and complete",),
        )
    if cancellation_check is not None:
        cancellation_check("cylindrical patch physical chart")
    chart = CylindricalMetricChart.from_geometry(geometry, patch.face.id)
    binding = CylindricalPatchBinding(geometry, selected, patch, ((patch.face_use, chart),))
    if cancellation_check is not None:
        cancellation_check("cylindrical patch preparation complete")
    # No caller code can change the owner after this last authoritative check.
    binding.validate()
    return binding
