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

from collections import Counter
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from anygeometry.model import GeometryModel

from ..errors import MeshError
from ..mesh import Mesh

__all__ = ["DEFAULT_BACKEND", "MeshBackend", "available_backends", "generate_mesh", "resolve_backend"]

DEFAULT_BACKEND = "auto"


@runtime_checkable
class MeshBackend(Protocol):
    """A callable that meshes a geometry model."""

    def __call__(self, geometry: GeometryModel, **options: Any) -> Mesh:
        """Return a mesh of ``geometry``."""


def _mapped_backend() -> MeshBackend:
    from ..mapped import generate_mesh as mapped_generate

    return mapped_generate


def _auto_backend() -> MeshBackend:
    from ..hybrid import generate_hybrid_mesh

    return generate_hybrid_mesh


def _native_backend() -> MeshBackend:
    from ..hybrid import MeshingStrategy, generate_hybrid_mesh

    def generate_native(geometry: GeometryModel, **options: Any) -> Mesh:
        options.setdefault("strategy", MeshingStrategy.NATIVE)
        return generate_hybrid_mesh(geometry, **options)

    return generate_native


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
    "auto": _auto_backend,
    "mapped": _mapped_backend,
    "native": _native_backend,
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

    ``auto`` is the production hybrid selector, ``native`` forces the native
    surface path, and ``mapped`` preserves the established mapped-only contract.
    Options are passed through because compatibility backends retain distinct
    feature sets rather than silently dropping unsupported controls.
    """

    key = str(backend).strip().lower()
    selected = resolve_backend(key)
    if key != "auto":
        return selected(geometry, **options)
    try:
        return selected(geometry, **options)
    except MeshError as primary:
        eligible, reason = _qualified_gmsh_fallback(geometry, options)
        if not eligible:
            raise
        try:
            fallback = _gmsh_backend()
        except MeshError:
            raise primary
        gmsh_options = {
            name: options[name]
            for name in ("target_size", "face_ids", "order", "recombine")
            if name in options
        }
        try:
            mesh = fallback(geometry, **gmsh_options)
        except MeshError as secondary:
            raise MeshError(
                f"automatic hybrid meshing rejected: {primary}; qualified "
                f"Gmsh fallback also rejected: {secondary}"
            ) from secondary
        mesh.geometry_model_id = geometry.model_id
        mesh.geometry_revision = geometry.revision
        mesh.hybrid_diagnostics["complex_geometry"] = {
            "classification": "planar_unshared_external_fallback",
            "attempted_strategies": ["hybrid", "gmsh"],
            "selected_strategy": "gmsh",
            "fallback_reason": str(primary),
            "qualification_reason": reason,
        }
        return mesh


def _qualified_gmsh_fallback(
    geometry: GeometryModel,
    options: Dict[str, Any],
) -> tuple[bool, str]:
    allowed = {"target_size", "face_ids", "order", "recombine"}
    if set(options).difference(allowed):
        return False, "hybrid-only controls are present"
    selected_faces = (
        tuple(int(face_id) for face_id in options["face_ids"])
        if options.get("face_ids") is not None
        else tuple(int(face_id) for face_id in geometry.faces)
    )
    if not selected_faces:
        return False, "no selected faces"
    edge_counts: Counter[int] = Counter()
    for face_id in selected_faces:
        face = geometry.faces.get(face_id)
        if face is None:
            return False, f"unknown face {face_id}"
        for boundary in (face.loop, *getattr(face, "holes", ())):
            edge_counts.update(int(item.edge) for item in boundary)
    if any(count > 1 for count in edge_counts.values()):
        return False, "selected faces share topology-owned edges"
    return True, "planar unshared faces with Gmsh-compatible controls"
