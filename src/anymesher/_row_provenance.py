"""Opt-in source identities carried from a material input to selected core rows."""

from dataclasses import dataclass

import numpy as np

from .errors import MeshError


def material_input_rows(receipt):
    """Map actual PSLG input rows to owner IDs, including repeated occurrences."""
    payload = receipt.to_dict()
    loops = payload["emitted_chart_loops"]
    pins = payload["pinned_inputs"]
    paths = payload["interior_paths"]
    boundaries = payload["boundary_inputs"]
    count = sum(len(loop) for loop in loops)
    if count != len(boundaries):
        raise MeshError("material input boundary token count differs from emitted rows")
    result = {index: int(item["global_node"]) for index, item in enumerate(boundaries)}
    offset = count
    for item in pins:
        result[offset] = int(item["global_node"])
        offset += 1
    constraint_count = sum(len(path["global_nodes"]) - 1 for path in paths)
    if constraint_count != len(payload["emitted_chart_constraints"]):
        raise MeshError("material input constraint token count differs from emitted rows")
    for path in paths:
        for first, second in zip(path["global_nodes"], path["global_nodes"][1:]):
            result[offset] = int(first)
            result[offset + 1] = int(second)
            offset += 2
    return result


def retry_input_rows(original, previous, boundary_count, interior_rows,
                     retry_interior_count, constraint_count):
    """Rebuild input indices from row lineage, never from chart coordinates."""
    if retry_interior_count < len(interior_rows):
        raise MeshError("material retry interior count omits added rows")
    previous_by_row = {int(row): int(node) for node, row in previous}
    result = {int(row): int(node) for row, node in original.items()
              if int(row) < boundary_count}
    for position, old_row in enumerate(interior_rows):
        node = previous_by_row.get(int(old_row))
        if node is not None:
            result[boundary_count + position] = node
    old_constraint_start = max(original, default=-1) + 1 - 2 * constraint_count
    new_constraint_start = boundary_count + retry_interior_count
    for index in range(2 * constraint_count):
        old = old_constraint_start + index
        if old in original:
            result[new_constraint_start + index] = int(original[old])
    return result


@dataclass(frozen=True)
class OutputRowProvenance:
    """Immutable row ledger: source ID or generated row, for one selected core."""

    source_by_row: tuple[int | None, ...]
    source_to_row: tuple[tuple[int, int], ...]

    def validate(self, core):
        if len(self.source_by_row) != core.num_nodes:
            raise MeshError("material output origin row count changed")
        active = set(map(int, core.triangle_connectivity[core.triangle_active].ravel()))
        active.update(map(int, core.quad_connectivity[core.quad_active].ravel()))
        seen = set()
        for node, row in self.source_to_row:
            if (node in seen or not 0 <= row < core.num_nodes
                    or row not in active or not core.node_active[row]
                    or self.source_by_row[row] != node):
                raise MeshError("material output lost or duplicated a protected source row")
            seen.add(node)
        if sum(node is not None for node in self.source_by_row) != len(seen):
            raise MeshError("material generated row acquired a source identity")
        return {row: node for node, row in self.source_to_row}


def bind_output_rows(core, candidate_points, triangulation, expected_ids,
                     expected_core_points):
    pairs = tuple((int(node), int(row)) for node, row in triangulation.protected_node_rows)
    if {node for node, _ in pairs} != set(map(int, expected_ids)):
        raise MeshError("material output protected ID set differs from input")
    points = np.asarray(candidate_points, dtype=np.float64)
    original = triangulation.points
    expected_core = np.asarray(expected_core_points, dtype=np.float64)
    if (expected_core.shape != (len(points), 3)
            or core.num_nodes < len(points)
            or not np.array_equal(core.node_coordinates[:len(points)], expected_core)):
        raise MeshError("material output reordered protected source rows")
    if len(points) < len(original) or any(
            not np.array_equal(points[row], original[row]) for _, row in pairs):
        raise MeshError("material output moved a protected source row")
    by_row = [None] * core.num_nodes
    for node, row in pairs:
        if (row < 0 or row >= len(by_row) or by_row[row] is not None
                or int(core.node_ids[row]) != row + 1):
            raise MeshError("material output has conflicting protected rows")
        by_row[row] = node
    record = OutputRowProvenance(tuple(by_row), tuple(sorted(pairs)))
    record.validate(core)
    core._output_row_provenance = record
    return record
