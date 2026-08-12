"""Revision-bound structural geometry views and lightweight change queues.

The mesher consumes an immutable *view* of an ANYgeometry document.  Keeping
the document identity, revision, and tolerance policy beside the structural
ownership indexes prevents a background job from quietly combining records
from different revisions.  Geometry records are immutable in ANYgeometry, so
copying the owner mappings is enough to make the snapshot stable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any

import numpy as np
from anygeometry.transactions import AABBChange, ChangeSet

from .errors import MeshError

__all__ = [
    "ChangeSetQueue",
    "GeometryMeshingView",
    "StaleMeshingViewError",
    "coalesce_change_sets",
]


class StaleMeshingViewError(MeshError):
    """Raised when a view is used after its geometry owner has advanced."""


def _readonly_copy(source: Mapping[int, Any]) -> Mapping[int, Any]:
    return MappingProxyType(dict(source))


class GeometryMeshingView:
    """A read-only structural snapshot bound to one model revision.

    The source model remains available only for numerical geometry evaluation.
    Every such call checks the binding first; ownership and topology reads use
    the copied mappings and therefore cannot observe a half-new revision.
    """

    def __init__(self, model: object) -> None:
        required = (
            "model_id",
            "revision",
            "tolerance",
            "vertices",
            "edges",
            "faces",
            "parts",
            "sheets",
            "face_uses",
            "coedges",
            "members",
            "member_edge_uses",
            "attachments",
            "junctions",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(
                "geometry model is missing meshing state: " + ", ".join(missing)
            )

        self._source = model
        self.model_id = getattr(model, "model_id")
        self.revision = int(getattr(model, "revision"))
        self.tolerance = getattr(model, "tolerance")

        self.vertices = _readonly_copy(getattr(model, "vertices"))
        self.edges = _readonly_copy(getattr(model, "edges"))
        self.faces = _readonly_copy(getattr(model, "faces"))
        self.parts = _readonly_copy(getattr(model, "parts"))
        self.sheets = _readonly_copy(getattr(model, "sheets"))
        self.face_uses = _readonly_copy(getattr(model, "face_uses"))
        self.coedges = _readonly_copy(getattr(model, "coedges"))
        self.members = _readonly_copy(getattr(model, "members"))
        self.member_edge_uses = _readonly_copy(
            getattr(model, "member_edge_uses")
        )
        self.attachments = _readonly_copy(getattr(model, "attachments"))
        self.junctions = _readonly_copy(getattr(model, "junctions"))

        sheet_faces: dict[int, tuple[int, ...]] = {}
        face_sheets: dict[int, set[int]] = {}
        edge_sheets: dict[int, set[int]] = {}
        face_use_sheet: dict[int, int] = {}
        for sheet_id, sheet in self.sheets.items():
            faces: list[int] = []
            for use_id in sheet.face_use_ids:
                use = self.face_uses.get(use_id)
                if use is None or use.sheet_id != sheet_id:
                    raise MeshError(
                        f"sheet {sheet_id} has invalid face-use ownership {use_id}"
                    )
                face_use_sheet[use_id] = sheet_id
                faces.append(int(use.face_id))
                face_sheets.setdefault(int(use.face_id), set()).add(sheet_id)
                for coedge_id in use.coedge_ids:
                    coedge = self.coedges.get(coedge_id)
                    if coedge is None or coedge.face_use_id != use_id:
                        raise MeshError(
                            f"face use {use_id} has invalid coedge ownership "
                            f"{coedge_id}"
                        )
                    edge_sheets.setdefault(int(coedge.edge_id), set()).add(sheet_id)
            sheet_faces[sheet_id] = tuple(faces)

        member_uses: dict[int, tuple[int, ...]] = {}
        edge_members: dict[int, set[int]] = {}
        for member_id, member in self.members.items():
            uses: list[int] = []
            for use_id in member.edge_use_ids:
                use = self.member_edge_uses.get(use_id)
                if use is None or use.member_id != member_id:
                    raise MeshError(
                        f"member {member_id} has invalid edge-use ownership {use_id}"
                    )
                uses.append(use_id)
                edge_members.setdefault(int(use.edge_id), set()).add(member_id)
            member_uses[member_id] = tuple(uses)

        self.sheet_faces = MappingProxyType(sheet_faces)
        self.member_uses = MappingProxyType(member_uses)
        self.face_sheet_owners = MappingProxyType(
            {key: tuple(sorted(value)) for key, value in face_sheets.items()}
        )
        self.edge_sheet_owners = MappingProxyType(
            {key: tuple(sorted(value)) for key, value in edge_sheets.items()}
        )
        self.edge_member_owners = MappingProxyType(
            {key: tuple(sorted(value)) for key, value in edge_members.items()}
        )
        self.face_use_sheet_owner = MappingProxyType(face_use_sheet)

    @classmethod
    def capture(cls, model: object) -> "GeometryMeshingView":
        """Explicit snapshot spelling for call sites that prefer it."""

        return cls(model)

    @property
    def source(self) -> object:
        """The numerical geometry owner, guarded by :meth:`assert_current`."""

        return self._source

    def is_current(self, model: object | None = None) -> bool:
        owner = self._source if model is None else model
        return (
            getattr(owner, "model_id", None) == self.model_id
            and getattr(owner, "revision", None) == self.revision
            and getattr(owner, "tolerance", None) == self.tolerance
        )

    def assert_current(self, model: object | None = None) -> None:
        owner = self._source if model is None else model
        if getattr(owner, "model_id", None) != self.model_id:
            raise StaleMeshingViewError("meshing view belongs to a different model")
        current = getattr(owner, "revision", None)
        if current != self.revision:
            raise StaleMeshingViewError(
                f"meshing view revision {self.revision} is stale; model is at "
                f"revision {current}"
            )
        if getattr(owner, "tolerance", None) != self.tolerance:
            raise StaleMeshingViewError("geometry tolerance changed after view capture")

    def faces_for_sheet(self, sheet_id: int) -> tuple[int, ...]:
        try:
            return self.sheet_faces[int(sheet_id)]
        except KeyError as error:
            raise MeshError(f"no sheet {sheet_id}") from error

    def edge_uses_for_member(self, member_id: int) -> tuple[object, ...]:
        try:
            identifiers = self.member_uses[int(member_id)]
        except KeyError as error:
            raise MeshError(f"no member {member_id}") from error
        return tuple(self.member_edge_uses[item] for item in identifiers)

    def sheets_for_face(self, face_id: int) -> tuple[int, ...]:
        return self.face_sheet_owners.get(int(face_id), ())

    def sheets_using_edge(self, edge_id: int) -> tuple[int, ...]:
        return self.edge_sheet_owners.get(int(edge_id), ())

    def members_using_edge(self, edge_id: int) -> tuple[int, ...]:
        return self.edge_member_owners.get(int(edge_id), ())

    def edge_point(self, edge_id: int, parameter: float) -> np.ndarray:
        """Evaluate one exact normalized edge parameter on the bound owner."""

        self.assert_current()
        made = float(parameter)
        if not np.isfinite(made) or not 0.0 <= made <= 1.0:
            raise MeshError("edge parameter must be finite and in [0, 1]")
        sample = getattr(self._source, "sample_edge")(
            int(edge_id), np.asarray((made,), dtype=float)
        )
        return np.asarray(sample[0], dtype=float)

    def edge_length(self, edge_id: int) -> float:
        self.assert_current()
        return float(getattr(self._source, "edge_length")(int(edge_id)))

    def closest_edge_point(
        self, edge_id: int, point: object
    ) -> tuple[np.ndarray, float, float]:
        self.assert_current()
        projected, parameter, distance = getattr(
            self._source, "closest_edge_point"
        )(int(edge_id), np.asarray(point, dtype=float))
        return np.asarray(projected, dtype=float), float(parameter), float(distance)

    def face_point(self, face_id: int, u: float, v: float) -> np.ndarray:
        self.assert_current()
        return np.asarray(
            getattr(self._source, "face_point")(int(face_id), float(u), float(v)),
            dtype=float,
        )

    def effective_length(self, extent: float = 0.0) -> float:
        return float(self.tolerance.effective_length(float(extent)))


def _coalesced_aabbs(changes: tuple[ChangeSet, ...]) -> tuple[AABBChange, ...]:
    values: dict[tuple[str, int], AABBChange] = {}
    for change in changes:
        for item in change.affected_aabbs:
            previous = values.get(item.entity)
            values[item.entity] = AABBChange(
                item.entity,
                item.before if previous is None else previous.before,
                item.after,
            )
    return tuple(values[key] for key in sorted(values))


def coalesce_change_sets(changes: Iterable[ChangeSet]) -> ChangeSet | None:
    """Fold contiguous committed changes into one deterministic delta."""

    made = tuple(changes)
    if not made:
        return None
    if any(not isinstance(item, ChangeSet) for item in made):
        raise TypeError("changes must contain ChangeSet values")
    for first, second in zip(made, made[1:]):
        if first.revision_after != second.revision_before:
            raise ValueError(
                "cannot coalesce non-contiguous revisions "
                f"{first.revision_after} and {second.revision_before}"
            )

    replacements: dict[object, tuple[object, ...]] = {}
    for change in made:
        for original, descendants in change.replacements:
            replacements[original] = descendants

    def keys(name: str) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(
                {
                    key
                    for change in made
                    for key in getattr(change, name)
                }
            )
        )

    replacement_items = tuple(
        (key, replacements[key])
        for key in sorted(
            replacements,
            key=lambda item: (getattr(item, "kind", ""), getattr(item, "id", 0)),
        )
    )
    return ChangeSet(
        revision_before=made[0].revision_before,
        revision_after=made[-1].revision_after,
        added=keys("added"),
        removed=keys("removed"),
        modified=keys("modified"),
        replacements=replacement_items,
        ownership_changes=keys("ownership_changes"),
        member_changes=keys("member_changes"),
        attachment_changes=keys("attachment_changes"),
        group_changes=tuple(
            sorted({name for change in made for name in change.group_changes})
        ),
        tag_changes=keys("tag_changes"),
        affected_aabbs=_coalesced_aabbs(made),
        invalidated_caches=keys("invalidated_caches"),
        spatial_updates=keys("spatial_updates"),
        feature_history_changed=any(
            item.feature_history_changed for item in made
        ),
        document_settings_changed=any(
            item.document_settings_changed for item in made
        ),
    )


class ChangeSetQueue:
    """Small thread-safe FIFO suitable for an ANYgeometry change hook."""

    def __init__(self, *, revision: int | None = None) -> None:
        self._items: deque[ChangeSet] = deque()
        self._last_revision = None if revision is None else int(revision)
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def last_revision(self) -> int | None:
        with self._lock:
            return self._last_revision

    def push(self, change: ChangeSet) -> None:
        if not isinstance(change, ChangeSet):
            raise TypeError("change must be a ChangeSet")
        with self._lock:
            expected = self._last_revision
            if expected is not None and change.revision_before != expected:
                raise ValueError(
                    f"change starts at revision {change.revision_before}; "
                    f"queue expected {expected}"
                )
            self._items.append(change)
            self._last_revision = change.revision_after

    enqueue = push
    __call__ = push

    def drain(self) -> tuple[ChangeSet, ...]:
        with self._lock:
            made = tuple(self._items)
            self._items.clear()
            return made

    def drain_coalesced(self) -> ChangeSet | None:
        return coalesce_change_sets(self.drain())

