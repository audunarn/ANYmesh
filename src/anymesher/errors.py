"""Exception types, in one place so nothing has to import a sibling for them.

``GeometryError`` used to live with the geometry model and ``MeshError`` with the
mapped mesher.  That worked until the chain-sampling helpers moved into the
geometry package: they raise ``MeshError``, and importing it from the mesher
would have recreated the import cycle the move was made to remove.

Both derive from ``ValueError``, because both mean "the input does not describe
something that can exist" rather than an internal failure.
"""

from __future__ import annotations

__all__ = ["GeometryError", "MeshError"]


class GeometryError(ValueError):
    """Raised when a modelling operation would produce invalid topology."""


class MeshError(ValueError):
    """Raised when a mesh cannot be generated from the given geometry."""
