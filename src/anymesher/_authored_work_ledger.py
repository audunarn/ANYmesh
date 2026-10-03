"""Carry the original native-mesh work allowance into detached root work.

This is accounting only. It cannot authorize a new meshing attempt, extra time,
or publication of a candidate mesh.
"""

from dataclasses import dataclass, replace
from numbers import Integral

from .errors import MeshError


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise MeshError(f"authored work {name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class AuthoredWorkLedger:
    insertion_limit: int
    topology_limit: int
    initial_insertions: int
    initial_operations: int
    further_insertions: int = 0
    further_operations: int = 0

    @property
    def remaining_insertions(self):
        return self.insertion_limit - self.initial_insertions - self.further_insertions

    @property
    def remaining_operations(self):
        return self.topology_limit - self.initial_operations - self.further_operations

    @classmethod
    def from_native_report(cls, options, report):
        """Reject missing/changed receipts instead of minting a fresh budget."""
        if not isinstance(report, dict):
            raise MeshError("authored work needs the original native report")
        names = ("selected_route", "cancelled", "insertion_budget",
                 "topology_budget", "insertions", "topology_operations",
                 "shared_segment_splits", "shared_nodes")
        if any(name not in report for name in names):
            raise MeshError("authored work report lacks original budget evidence")
        if (not isinstance(report["selected_route"], str)
                or not report["selected_route"].startswith("frontal_delaunay")
                or report["cancelled"] is not False):
            raise MeshError("authored work needs a completed native report")
        insertion_limit = _count(report["insertion_budget"], "insertion budget")
        topology_limit = _count(report["topology_budget"], "topology budget")
        if (insertion_limit != options.max_insertions
                or topology_limit != options.max_topology_operations):
            raise MeshError("authored work report differs from original options")
        insertions = _count(report["insertions"], "insertions")
        operations = _count(report["topology_operations"], "operations")
        splits = _count(report["shared_segment_splits"], "shared splits")
        if not isinstance(report["shared_nodes"], list):
            raise MeshError("authored work shared node receipt is invalid")
        if "reserved_node_reuses" in report or "staged_point_insertions" in report:
            if "reserved_node_reuses" not in report or "staged_point_insertions" not in report:
                raise MeshError("authored work shared-node accounting is incomplete")
            reserved = _count(report["reserved_node_reuses"], "reserved reuses")
            if _count(report["staged_point_insertions"], "staged insertions") != insertions + reserved:
                raise MeshError("authored work shared-node accounting changed")
        elif splits or report["shared_nodes"]:
            raise MeshError("authored work lacks shared-node budget evidence")
        else:
            reserved = 0
        if insertions + reserved > insertion_limit or operations > topology_limit:
            raise MeshError("authored work original operation exceeded its budget")
        return cls(insertion_limit, topology_limit, insertions + reserved, operations)

    def charge(self, *, insertions, operations):
        made = replace(self,
            further_insertions=self.further_insertions + _count(insertions, "new insertions"),
            further_operations=self.further_operations + _count(operations, "new operations"))
        if made.remaining_insertions < 0 or made.remaining_operations < 0:
            raise MeshError("authored work would exceed the original operation budget")
        return made
