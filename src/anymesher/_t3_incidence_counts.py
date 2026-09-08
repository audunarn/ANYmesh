"""Immutable, independently established incidence-count proof snapshots."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._t3_edge_membership import EdgeMembershipMap
from .errors import MeshError


def _validate(edge, count, *, allow_zero):
    if (not isinstance(edge, tuple) or len(edge) != 2
            or any(type(node) is not int for node in edge)
            or edge[0] < 0 or edge[0] >= edge[1]
            or type(count) is not int
            or not (0 if allow_zero else 1) <= count <= 2):
        raise MeshError("qualified incidence count snapshot is invalid")


@dataclass(frozen=True, slots=True)
class IncidenceCounts(Mapping):
    """No backing incidence map or mutable dictionary is shared with a proof."""
    _root: object
    _size: int

    get = EdgeMembershipMap.get
    __getitem__ = EdgeMembershipMap.__getitem__
    __contains__ = EdgeMembershipMap.__contains__
    __iter__ = EdgeMembershipMap.__iter__
    __len__ = EdgeMembershipMap.__len__

    @classmethod
    def from_mapping(cls, values):
        if type(values) is cls:
            return values
        items = list(values.items())
        for edge, count in items:
            _validate(edge, count, allow_zero=False)
        staged = EdgeMembershipMap.from_items(items)
        return cls(staged._root, len(staged))

    def with_updates(self, values):
        staged = EdgeMembershipMap()
        staged._root = self._root
        staged._size = self._size
        for edge, count in values.items():
            _validate(edge, count, allow_zero=True)
            if count:
                staged[edge] = count
            elif edge in staged:
                del staged[edge]
        return type(self)(staged._root, len(staged))

