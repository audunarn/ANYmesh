"""Incremental element activity and exposed shell-boundary tracking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import MeshError
from .mesh import Mesh
from .mesh_bvh import MeshElementBVH

__all__ = [
    "ActivityUpdate",
    "BoundaryFacet",
    "DamageState",
    "ElementActivity",
]

SideKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BoundaryFacet:
    key: SideKey
    node_ids: tuple[int, int]
    active_elements: tuple[int, ...]
    inactive_elements: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActivityUpdate:
    activated: tuple[int, ...]
    deactivated: tuple[int, ...]
    affected_elements: tuple[int, ...]
    affected_sides: tuple[SideKey, ...]
    newly_exposed: tuple[BoundaryFacet, ...]
    no_longer_exposed: tuple[BoundaryFacet, ...]

    @property
    def changed_elements(self) -> tuple[int, ...]:
        return tuple(sorted((*self.activated, *self.deactivated)))


class ElementActivity:
    """Mutable activity mask with side-local exposed-boundary maintenance."""

    def __init__(
        self, mesh: Mesh, *, active_elements: Iterable[int] | None = None
    ) -> None:
        self.mesh = mesh
        shell_ids = set(mesh.shells)
        all_ids = shell_ids | set(mesh.beams)
        self._all = all_ids
        self._active = set(all_ids) if active_elements is None else {int(item) for item in active_elements}
        unknown = self._active - all_ids
        if unknown:
            raise MeshError(f"activity references unknown elements {sorted(unknown)}")

        sides_by_element: dict[int, tuple[SideKey, ...]] = {}
        incidence: dict[SideKey, list[tuple[int, tuple[int, int]]]] = {}
        for element_id in sorted(shell_ids):
            corners = mesh.corners_of(element_id)
            oriented = tuple(zip(corners, corners[1:] + corners[:1]))
            keys: list[SideKey] = []
            for first, second in oriented:
                key = (min(int(first), int(second)), max(int(first), int(second)))
                keys.append(key)
                incidence.setdefault(key, []).append(
                    (int(element_id), (int(first), int(second)))
                )
            sides_by_element[element_id] = tuple(keys)
        self._sides_by_element = sides_by_element
        self._incidence = {key: tuple(value) for key, value in incidence.items()}
        self._exposed: dict[SideKey, BoundaryFacet] = {}
        for key in self._incidence:
            facet = self._facet(key)
            if facet is not None:
                self._exposed[key] = facet

    @property
    def active_elements(self) -> frozenset[int]:
        return frozenset(self._active)

    @property
    def inactive_elements(self) -> frozenset[int]:
        return frozenset(self._all - self._active)

    @property
    def exposed_boundaries(self) -> Mapping[SideKey, BoundaryFacet]:
        return MappingProxyType(dict(self._exposed))

    def _facet(self, key: SideKey) -> BoundaryFacet | None:
        incidence = self._incidence[key]
        active = tuple(sorted(item[0] for item in incidence if item[0] in self._active))
        inactive = tuple(sorted(item[0] for item in incidence if item[0] not in self._active))
        exposed = bool(active) and (len(incidence) == 1 or bool(inactive))
        if not exposed:
            return None
        owner = next(item for item in incidence if item[0] in self._active)
        return BoundaryFacet(key, owner[1], active, inactive)

    def set_activity(
        self, element_ids: Iterable[int], *, active: bool
    ) -> ActivityUpdate:
        requested = {int(item) for item in element_ids}
        unknown = requested - self._all
        if unknown:
            raise MeshError(f"activity references unknown elements {sorted(unknown)}")
        changed = requested - self._active if active else requested & self._active
        if not changed:
            return ActivityUpdate((), (), (), (), (), ())
        affected_sides = {
            side for element in changed for side in self._sides_by_element.get(element, ())
        }
        before = {key: self._exposed.get(key) for key in affected_sides}
        if active:
            self._active.update(changed)
        else:
            self._active.difference_update(changed)
        affected_elements = {
            item[0]
            for side in affected_sides
            for item in self._incidence[side]
        } | changed
        newly: list[BoundaryFacet] = []
        removed: list[BoundaryFacet] = []
        for key in sorted(affected_sides):
            old = before[key]
            new = self._facet(key)
            if new is None:
                self._exposed.pop(key, None)
            else:
                self._exposed[key] = new
            if old is None and new is not None:
                newly.append(new)
            elif old is not None and new is None:
                removed.append(old)
            elif old is not None and new is not None and old != new:
                removed.append(old)
                newly.append(new)
        return ActivityUpdate(
            tuple(sorted(changed)) if active else (),
            () if active else tuple(sorted(changed)),
            tuple(sorted(affected_elements)),
            tuple(sorted(affected_sides)),
            tuple(newly),
            tuple(removed),
        )

    def activate(self, element_ids: Iterable[int]) -> ActivityUpdate:
        return self.set_activity(element_ids, active=True)

    def deactivate(self, element_ids: Iterable[int]) -> ActivityUpdate:
        return self.set_activity(element_ids, active=False)

    def sync_bvh(self, bvh: MeshElementBVH) -> None:
        """Update only the BVH active mask; its static tree remains reusable."""

        bvh.replace_active(set(self._active).intersection(bvh.element_ids))


DamageState = ElementActivity
