"""General cylinder trims certified by ANYgeometry's material chart owner."""
from dataclasses import dataclass

from anygeometry import query_trimmed_surface_charts, validate_trimmed_surface_charts_binding
from ._cylindrical_chart import CylindricalMetricChart
from .errors import MeshError


def _owner(geometry):
    guard=getattr(geometry,"assert_current",None)
    if callable(guard):
        guard()
        return geometry.source
    return geometry


@dataclass(frozen=True,slots=True)
class TrimmedCylinderBinding:
    geometry: object
    result: object
    charts: tuple

    @property
    def model_id(self):
        return self.result.model_id

    @property
    def revision(self):
        return self.result.revision

    @property
    def face_records(self):
        return self.result.charts

    @property
    def certification_kind(self):
        return "general_analytic_trimmed_material_charts"

    def validate(self,*,cancellation_check=None):
        validate_trimmed_surface_charts_binding(_owner(self.geometry),self.result,
                                                cancellation_check=cancellation_check)

    def chart_for(self,face_use,*,cancellation_check=None):
        self.validate(cancellation_check=cancellation_check)
        for selected,chart in self.charts:
            if selected==face_use:
                chart._current()
                if chart.support_view is not None:
                    # Native/frontal consumers pair this chart with face-frame
                    # uv; MESH-B A2 admits the support frame on quad-first only.
                    raise MeshError("a support-frame cylinder chart (distinct face parameterization) "
                                    "is admitted only on the quad-first route")
                return chart
        raise MeshError("FaceUse is not selected in the general cylinder binding")


def prepare_trimmed_cylinders(geometry,face_uses,*,cancellation_check=None,
                              support_frame_parameterizations=False):
    owner=_owner(geometry)
    result=query_trimmed_surface_charts(owner,face_uses,expected_revision=owner.revision,
                                       cancellation_check=cancellation_check)
    charts=tuple((record.face_use,CylindricalMetricChart.from_geometry(
                       geometry,record.face.id,
                       support_frame_parameterization=support_frame_parameterizations))
                 for record in result.charts)
    binding=TrimmedCylinderBinding(geometry,result,charts)
    binding.validate(cancellation_check=cancellation_check)
    return binding
