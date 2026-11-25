"""Meshing errors plus the historical geometry-error import path."""

from __future__ import annotations

from anygeometry.errors import GeometryError

__all__ = [
    "GeometryError", "MeshError", "StructuredQualityRejected", "NativeSurfaceUnsupported",
    "StructuredElementLimitExceeded",
]

class MeshError(ValueError):
    """Raised when a mesh cannot be generated from the given geometry."""


class StructuredQualityRejected(MeshError):
    """A completed structured candidate failed its unchanged quality gate."""


class NativeSurfaceUnsupported(MeshError):
    """The requested native placement has no qualified chart for this support."""


class StructuredElementLimitExceeded(MeshError):
    """The exact global seed solution needs more structured elements than the policy allows.

    Raised only from the solved seed counts (never from an area estimate), so a caller may treat it as an exact,
    route-specific refusal. ``estimated`` and ``maximum`` are the numbers in the unchanged message.
    """

    def __init__(self, estimated: int, maximum: int) -> None:
        self.estimated = int(estimated)
        self.maximum = int(maximum)
        super().__init__(
            f"global seed solution estimates {self.estimated} structured elements, exceeding "
            f"maximum_estimated_elements={self.maximum}"
        )

    def __reduce__(self):
        return type(self), (self.estimated, self.maximum)
