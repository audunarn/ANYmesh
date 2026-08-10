"""What a mesh backend is, and how one is chosen.

A backend turns a geometry model into a :class:`~anymesher.mesh.Mesh`.  The
built-in mapped mesher is one; gmsh is another.  Both return the same container,
so everything downstream -- quality metrics, association lookups, export --
works the same either way.

They are not interchangeable in what they *guarantee*, and the difference is
recorded rather than smoothed over.  The mapped mesher produces a structured grid
per face and conformity by construction; gmsh produces an unstructured mesh with
no grid, so ``grid_of_face`` stays empty and anything that wanted a structured
index has to say so.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from anygeometry.model import GeometryModel

from ..errors import MeshError
from ..mesh import Mesh

__all__ = ["DEFAULT_BACKEND", "MeshBackend", "available_backends", "generate_mesh", "resolve_backend"]

DEFAULT_BACKEND = "mapped"


@runtime_checkable
class MeshBackend(Protocol):
    """A callable that meshes a geometry model."""

    def __call__(self, geometry: GeometryModel, **options: Any) -> Mesh:
        """Return a mesh of ``geometry``."""


def _mapped_backend() -> MeshBackend:
    from ..mapped import generate_mesh as mapped_generate

    return mapped_generate


def _gmsh_backend() -> MeshBackend:
    try:
        from .gmsh import generate_mesh as gmsh_generate
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise MeshError(
            "the gmsh backend needs the gmsh package: pip install 'ANYmesher[gmsh]'"
        ) from error
    return gmsh_generate


# Loaded lazily, so importing anymesher never imports gmsh.
_BACKEND_LOADERS: Dict[str, Any] = {
    "mapped": _mapped_backend,
    "gmsh": _gmsh_backend,
}


def available_backends() -> Tuple[str, ...]:
    """Every backend name, whether or not its dependency is installed.

    Names are listed unconditionally so an unavailable backend produces an
    explanation from :func:`resolve_backend` rather than looking like a typo.
    """

    return tuple(sorted(_BACKEND_LOADERS))


def resolve_backend(name: str) -> MeshBackend:
    """Return the backend callable for a name."""

    key = str(name).strip().lower()
    loader = _BACKEND_LOADERS.get(key)
    if loader is None:
        raise MeshError(
            f"unknown mesh backend {name!r}; expected one of {', '.join(available_backends())}"
        )
    return loader()


def generate_mesh(
    geometry: GeometryModel, *, backend: str = DEFAULT_BACKEND, **options: Any
) -> Mesh:
    """Mesh a geometry model with the named backend.

    Options are passed through, because the two backends do not take the same
    ones: the mapped mesher accepts seeding overrides and per-edge beam offsets,
    gmsh accepts its own algorithm choices, and pretending they share a signature
    would mean silently dropping whatever did not fit.
    """

    return resolve_backend(backend)(geometry, **options)
