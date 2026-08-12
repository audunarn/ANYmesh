"""Mesh backends: production hybrid/native, mapped compatibility, and gmsh."""

from __future__ import annotations

from .base import (
    DEFAULT_BACKEND,
    MeshBackend,
    available_backends,
    generate_mesh,
    resolve_backend,
)

__all__ = [
    "DEFAULT_BACKEND",
    "MeshBackend",
    "available_backends",
    "generate_mesh",
    "resolve_backend",
]
