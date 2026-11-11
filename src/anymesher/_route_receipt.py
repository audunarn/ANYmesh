"""Result-level route receipt for ``strategy='auto'`` on NURBS faces.

Immutable values only. ``TensorRefusal`` is the typed reason; the free-text ``detail`` is for people and is never
branched on.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TensorRefusal(Enum):
    REQUEST_OUTSIDE_ALLOWLIST = 'request_outside_allowlist'
    NO_OWNER_BOUND_REGION = 'no_owner_bound_region'
    NOT_LINEAR = 'not_linear'
    DECLARED_JUNCTION = 'declared_junction'
    FINER_RESOLUTION_REQUIRED = 'finer_resolution_required'
    TRIANGLES_REQUESTED = 'triangles_requested'
    NON_UNIFORM_SIZING = 'non_uniform_sizing'
    UNSUPPORTED_FACE = 'unsupported_face'  # ParametricTensorUnsupported from the tensor core
    QUALITY_OR_GROWTH_GATE = 'quality_or_growth_gate'


@dataclass(frozen=True)
class FaceRouteReceipt:
    face_id: int
    route_tried: str  # 'tensor'
    outcome: str  # 'tensor' or 'native_fallback'
    refusal: TensorRefusal | None
    detail: str
    work: tuple[tuple[str, int], ...]  # counters of the route that produced the face (tensor: cells, nodes)


@dataclass(frozen=True)
class NurbsRouteReceipt:
    faces: tuple[FaceRouteReceipt, ...]


def build_receipt(diagnostics_by_face) -> NurbsRouteReceipt | None:
    entries = []
    for face_id in sorted(diagnostics_by_face):
        d = diagnostics_by_face[face_id]
        if not isinstance(d, dict):
            continue
        kind = d.get('parametric_tensor_refusal_kind')
        tensor = d.get('parametric_tensor')
        if kind is not None:
            entries.append(FaceRouteReceipt(int(face_id), 'tensor', 'native_fallback', TensorRefusal(kind),
                                            str(d.get('parametric_tensor_refusal', '')), ()))
        elif tensor is not None:
            entries.append(FaceRouteReceipt(int(face_id), 'tensor', 'tensor', None, '',
                                            (('cells', int(tensor['cells'])), ('nodes', int(tensor['nodes'])))))
    return NurbsRouteReceipt(tuple(entries)) if entries else None
