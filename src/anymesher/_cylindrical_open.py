"""Read-only consumption of the owner-qualified open cylinder component."""

from dataclasses import dataclass

from anygeometry import (
    CylinderOpenComponentError,
    CylinderOpenComponentErrorCode,
    CylinderPatchStatus,
    evaluate_cylinder_open_component_occurrences,
    query_cylinder_open_component,
    validate_cylinder_open_component_binding,
)

from ._cylindrical_atlas import _owner
from ._cylindrical_chart import CylindricalMetricChart
from .errors import MeshError


@dataclass(frozen=True, slots=True)
class CylindricalOpenBinding:
    geometry: object
    face_uses: tuple
    component: object
    charts: tuple

    @property
    def model_id(self):
        return self.component.model_id

    @property
    def revision(self):
        return self.component.revision

    @property
    def face_records(self):
        return self.component.patches

    @property
    def certification_kind(self):
        return "nonperiodic_open_sector_component"

    def validate(self, *, cancellation_check=None):
        validate_cylinder_open_component_binding(
            _owner(self.geometry), self.component, self.face_uses,
            expected_revision=self.component.revision,
            cancellation_check=cancellation_check,
        )

    def evaluate(self, requests, *, cancellation_check=None):
        return evaluate_cylinder_open_component_occurrences(
            _owner(self.geometry), self.component, requests,
            face_uses=self.face_uses, expected_revision=self.component.revision,
            cancellation_check=cancellation_check,
        )

    def chart_for(self, face_use, *, cancellation_check=None):
        self.validate(cancellation_check=cancellation_check)
        for selected, chart in self.charts:
            if selected == face_use:
                chart._current()
                return chart
        raise MeshError("face use is not selected in this open cylinder component")


def prepare_cylindrical_open(geometry, face_uses, *, cancellation_check=None):
    owner = _owner(geometry)
    component = query_cylinder_open_component(
        owner, face_uses, expected_revision=owner.revision,
        cancellation_check=cancellation_check,
    )
    selected = component.requested_face_uses
    if component.status is not CylinderPatchStatus.QUALIFIED or not component.complete:
        raise CylinderOpenComponentError(
            CylinderOpenComponentErrorCode.UNQUALIFIED_RESULT,
            "; ".join(component.diagnostics) or "open component is not qualified",
        )
    charts = []
    for patch in component.patches:
        if cancellation_check is not None:
            cancellation_check("open cylinder physical sector chart")
        charts.append((patch.face_use, CylindricalMetricChart.from_geometry(geometry, patch.face.id)))
    binding = CylindricalOpenBinding(geometry, selected, component, tuple(charts))
    binding.validate()
    return binding
