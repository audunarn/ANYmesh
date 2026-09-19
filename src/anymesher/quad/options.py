"""Strict, versioned quad-first meshing options.

``QuadMeshingOptions`` is a separate, additive controls block for the quad-first
route. It is intentionally independent of :class:`anymesher.native_v2.NativeMeshingOptions`
(the legacy mapped/native path and its schema remain untouched).  Passing
``quad_options=None`` at an entry point executes the old path; an instance of
this class is the only signal that the quad-first contract applies.

The schema is frozen at Q0: a fixed field set, a strict schema tag, validated
enums and positive-integer resource guards, and a deterministic canonical
digest so callers can pin exact quad behaviour without coupling to internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Any, Mapping

from ..errors import MeshError

__all__ = ["QuadMeshingOptions", "QUAD_MESHING_OPTIONS_SCHEMA"]

QUAD_MESHING_OPTIONS_SCHEMA = "anymesher.quad-meshing-options/1"

_SEED_MODES = {"cdt"}
_ORIENTATIONS = {"cross_4theta", "boundary_tangent"}
_QUALITY_MODELS = {"shape", "shape_jacobian"}
_LINE_SEARCH_STRATEGIES = {"off", "safeguarded"}


def _positive_int(value: Any, *, minimum: int = 1, name: str = "") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        bound = "a positive integer" if minimum == 1 else f"an integer >= {minimum}"
        raise MeshError(f"{name} must be {bound}")
    return int(value)


@dataclass(frozen=True)
class QuadMeshingOptions:
    """Frozen quad-first controls; the legacy/native path stays untouched.

    Defaults are conservative and deterministic.  Enum-valued fields are only
    ever drawn from the Q0-frozen sets, and the three integer fields are
    resource guards that bound the front work, local optimization depth and
    cancellation granularity.
    """

    seed_mode: str = "cdt"
    orientation: str = "cross_4theta"
    quality_model: str = "shape"
    line_search: str = "safeguarded"
    max_front_iterations: int = 100_000
    max_local_optimizations: int = 64
    cancellation_interval: int = 256

    def __post_init__(self) -> None:
        if self.seed_mode not in _SEED_MODES:
            raise MeshError("seed_mode must be 'cdt'")
        if self.orientation not in _ORIENTATIONS:
            raise MeshError("orientation must be 'cross_4theta' or 'boundary_tangent'")
        if self.quality_model not in _QUALITY_MODELS:
            raise MeshError("quality_model must be 'shape' or 'shape_jacobian'")
        if self.line_search not in _LINE_SEARCH_STRATEGIES:
            raise MeshError("line_search must be 'off' or 'safeguarded'")
        # Guard resources; normalize ints (e.g. numpy scalars) to built-in int.
        object.__setattr__(
            self,
            "max_front_iterations",
            _positive_int(self.max_front_iterations, name="max_front_iterations"),
        )
        object.__setattr__(
            self,
            "max_local_optimizations",
            _positive_int(self.max_local_optimizations, minimum=0, name="max_local_optimizations"),
        )
        object.__setattr__(
            self,
            "cancellation_interval",
            _positive_int(self.cancellation_interval, name="cancellation_interval"),
        )
        if self.line_search == "off" and self.max_local_optimizations != 0:
            raise MeshError("line_search='off' requires max_local_optimizations=0")

    @classmethod
    def coerce(cls, value: "QuadMeshingOptions | Mapping[str, Any] | None") -> "QuadMeshingOptions | None":
        """Return ``None`` for the legacy path, else a validated instance.

        This is the single entry point used by callers distinguishing
        "old path" (``None``) from "quad-first" (an instance or mapping).
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise MeshError("quad_options must be None, QuadMeshingOptions or a mapping")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUAD_MESHING_OPTIONS_SCHEMA,
            "seed_mode": self.seed_mode,
            "orientation": self.orientation,
            "quality_model": self.quality_model,
            "line_search": self.line_search,
            "max_front_iterations": self.max_front_iterations,
            "max_local_optimizations": self.max_local_optimizations,
            "cancellation_interval": self.cancellation_interval,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QuadMeshingOptions":
        expected = {
            "schema", "seed_mode", "orientation", "quality_model", "line_search",
            "max_front_iterations", "max_local_optimizations", "cancellation_interval",
        }
        if set(raw) != expected:
            raise MeshError("quad meshing options contain an unknown or missing field")
        if raw.get("schema") != QUAD_MESHING_OPTIONS_SCHEMA:
            raise MeshError("quad meshing options schema mismatch")
        return cls(
            seed_mode=raw["seed_mode"],
            orientation=raw["orientation"],
            quality_model=raw["quality_model"],
            line_search=raw["line_search"],
            max_front_iterations=raw["max_front_iterations"],
            max_local_optimizations=raw["max_local_optimizations"],
            cancellation_interval=raw["cancellation_interval"],
        )

    @property
    def canonical_digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()
