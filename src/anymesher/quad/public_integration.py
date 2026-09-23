"""Q6 public quad-first integration surface.

Narrow public contract for exposing the quad-first route.  ``None`` remains the
legacy dispatch sentinel; an explicit :class:`QuadMeshingOptions` (or mapping)
is the only signal that quad-first behaviour applies.  Scopes the route does
not yet support are rejected with a typed :class:`QuadPublicUnsupported`
instead of falling back to the legacy path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any, Callable, Mapping, Optional

from ..errors import MeshError
from .options import QuadMeshingOptions
from .quad_mcf_worker import WorkerNotFound, find_worker as find_mcf_worker
from .quad_tinyad_worker import WorkerNotFoundQ5, find_worker as find_q5_worker

__all__ = [
    "QuadPublicUnsupported",
    "QuadCapabilityMissing",
    "QuadCapabilityReport",
    "advertise_quad_capabilities",
    "canonical_interface_edge",
    "coerce_public_quad_options",
    "publish_atomically",
    "route_quad_first",
]


class QuadPublicUnsupported(MeshError):
    """A requested public quad-first scope is not supported on this route."""


class QuadCapabilityMissing(MeshError):
    """An explicitly requested quad-first capability is not available."""


def canonical_interface_edge(a: int, b: int) -> tuple[int, int]:
    """Return the canonical sorted ``(min, max)`` form of an interface edge.

    Expects built-in integer entity IDs; ``a == b`` and non-int/bool values are
    rejected.
    """
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise MeshError(f"canonical_interface_edge {name} must be an integer ID")
    ia, ib = int(a), int(b)
    if ia == ib:
        raise MeshError("canonical_interface_edge requires two distinct entity IDs")
    return (ia, ib) if ia < ib else (ib, ia)


@dataclass(frozen=True)
class QuadCapabilityReport:
    """Capability advertisement for the public quad-first route."""

    quad_first_api: str
    compiled_native_support: bool
    front_path: str
    q4_count_worker: str
    q5_tinyad_worker: str
    mixed_q4_s3: bool
    unsupported_scope: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unsupported_scope", tuple(self.unsupported_scope))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _capability_field_names()}


def _capability_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(QuadCapabilityReport))


def coerce_public_quad_options(
    value: "QuadMeshingOptions | Mapping[str, Any] | None",
    *,
    order: str = "linear",
    planar: bool = True,
) -> "QuadMeshingOptions | None":
    """Validate the public quad-first option contract.

    ``None`` passes through unchanged as the legacy dispatch sentinel.  An
    explicit option outside the supported scope (``planar is False``, or an
    ``order`` other than ``'linear'``/``'quadratic'``) raises
    :class:`QuadPublicUnsupported` with no fallback.  ``order='quadratic'``
    is admitted only for the planar explicit scope; the quad-first result is
    then promoted in place to Q8/T6 after the linear topology is qualified.
    """
    if order not in ("linear", "quadratic") or not (type(planar) is bool and planar):
        if value is None:
            return None
        raise QuadPublicUnsupported(
            "quad-first public route requires planar=True and "
            "order='linear' or order='quadratic'"
        )
    return QuadMeshingOptions.coerce(value)


def publish_atomically(
    provisional: Any,
    *,
    validate: Callable[[Any], None],
    cancellation_check: Optional[Callable[[str], None]] = None,
    publish: Callable[[Any], Any],
) -> Any:
    """Validate then publish a provisional quad-first result.

    Order: ``validate(provisional)``, then
    ``cancellation_check('quad-first:before-publication')`` when provided, then
    ``publish(provisional)``.  If validation or the cancellation check raises,
    ``publish`` is never called and the exception propagates.
    """
    validate(provisional)
    if cancellation_check is not None:
        cancellation_check("quad-first:before-publication")
    return publish(provisional)


_MIXED_Q4_S3_ROUTE_SUPPORTED = True


def _s3_production_available() -> bool:
    """Return whether the qualified S3 production bridge is importable."""
    try:
        from ..s3_production import prepare_qualified_s3_mesh
    except ImportError:
        return False
    return callable(prepare_qualified_s3_mesh)


def advertise_quad_capabilities(
    *,
    mcf_worker_path: "str | os.PathLike[str] | None" = None,
    q5_worker_path: "str | os.PathLike[str] | None" = None,
    unsupported_scope: "str | tuple[str, ...]" = ("curved", "Q9+"),
) -> QuadCapabilityReport:
    """Build the :class:`QuadCapabilityReport` for the public route.

    Probe both worker binaries (Q4 count/MCF and Q5 TinyAD) through their
    existing adapters; a missing binary raises :class:`QuadCapabilityMissing`
    (a real failure, never a silent fallback).  ``compiled_native_support``
    is True when both binaries are located.
    """
    unsupported = (
        (unsupported_scope,)
        if isinstance(unsupported_scope, str) and unsupported_scope
        else tuple(unsupported_scope)
    )
    try:
        mcf = find_mcf_worker(mcf_worker_path)
    except WorkerNotFound as error:
        raise QuadCapabilityMissing(str(error)) from error
    try:
        q5 = find_q5_worker(q5_worker_path)
    except WorkerNotFoundQ5 as error:
        raise QuadCapabilityMissing(str(error)) from error
    return QuadCapabilityReport(
        quad_first_api="public/1",
        compiled_native_support=True,
        front_path="advancing_front",
        q4_count_worker=str(mcf),
        q5_tinyad_worker=str(q5),
        mixed_q4_s3=bool(_MIXED_Q4_S3_ROUTE_SUPPORTED and _s3_production_available()),
        unsupported_scope=unsupported,
    )


def route_quad_first(
    options: "QuadMeshingOptions | None",
    *,
    order: str = "linear",
    planar: bool = True,
    mcf_worker_path: "str | os.PathLike[str] | None" = None,
    q5_worker_path: "str | os.PathLike[str] | None" = None,
    require_workers: bool = True,
) -> "tuple[QuadMeshingOptions, QuadCapabilityReport]":
    """Validate the request and advertise public quad-first capabilities.

    Order matters: scope rejection (order/planar) happens first, so an
    out-of-scope explicit request raises :class:`QuadPublicUnsupported`
    *before* the worker probes run and *without* raising
    :class:`QuadCapabilityMissing`.  ``None`` short-circuits to
    ``(None, None)`` — the legacy dispatch sentinel.

    ``require_workers=True`` (the default) preserves the historical
    advertise-and-fail capability contract: a missing worker binary raises
    :class:`QuadCapabilityMissing`.  The PQ-M1 public face route executes
    self-contained and instead calls with ``require_workers=False``, which
    returns a truthful report (``compiled_native_support=False``, both workers
    ``NOT_INTEGRATED``) without probing.
    """
    if options is None:
        return None, None
    normalized = coerce_public_quad_options(options, order=order, planar=planar)
    if normalized is None:
        raise MeshError("explicit quad_options must not coerce to None")
    if require_workers:
        return normalized, advertise_quad_capabilities(
            mcf_worker_path=mcf_worker_path,
            q5_worker_path=q5_worker_path,
        )
    # PQ-M1 public execution is self-contained: the historical Q4 MCF and Q5
    # TinyAD adapters remain directly probeable via advertise_quad_capabilities,
    # but they are no longer required by the genuine planar front dataflow.
    return normalized, QuadCapabilityReport(
        quad_first_api="public/1",
        compiled_native_support=False,
        front_path="advancing_front",
        q4_count_worker="NOT_INTEGRATED",
        q5_tinyad_worker="NOT_INTEGRATED",
        mixed_q4_s3=bool(_MIXED_Q4_S3_ROUTE_SUPPORTED and _s3_production_available()),
        unsupported_scope=("curved", "Q9+"),
    )
