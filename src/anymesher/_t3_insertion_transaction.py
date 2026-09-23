"""Local proof for publishing one point insertion into validated T3 topology."""
from __future__ import annotations

import numpy as np

from .errors import MeshError
from ._t3_insertion_result import _edges_and_counts, _positive_cells, _source_rows


def validate_insertion_transaction(
    points, cells, node_owners, cell_owners, old_points, old_cells,
    old_node_owners, old_cell_owners, old_index, protected, splittable,
    report, *, owner, orientation_oracle, cancellation_check=None,
):
    """Check only changed-cell geometry against the previously validated state.

    Exact vector row matching still scans connectivity; it is not a location
    cache or a claim of constant-time insertion. Source incidence is reused,
    never reconstructed. No passed array or index is modified.
    """
    def check(phase):
        if cancellation_check is not None:
            cancellation_check("native-v2 insertion cavity " + phase)

    check("validation")
    inserted = len(old_points)
    if (points.shape != (inserted + 1, 2)
            or not np.array_equal(points[:-1].view(np.uint64),
                                  old_points.view(np.uint64))
            or not np.all(np.isfinite(points[-1]))):
        raise MeshError("insertion transaction changed retained point coordinates")
    if (node_owners.shape != (len(points),)
            or cell_owners.shape != (len(cells),)
            or not np.array_equal(node_owners[:-1], old_node_owners)):
        raise MeshError("insertion transaction changed retained ownership")
    if cells.ndim != 2 or cells.shape[1] != 3 or cells.dtype.kind not in "iu":
        raise MeshError("insertion transaction connectivity is malformed")
    source = _source_rows(old_cells, cells)
    retained = source >= 0
    old_rows = source[retained]
    if len(np.unique(old_rows)) != len(old_rows):
        raise MeshError("insertion transaction repeats a retained cell")
    old_kept = np.zeros(len(old_cells), dtype=bool)
    old_kept[old_rows] = True
    removed = old_cells[~old_kept]
    added = cells[~retained]
    if (not len(removed) or not len(added)
            or type(report.get("removed_triangles")) is not int
            or type(report.get("added_triangles")) is not int
            or len(removed) != report["removed_triangles"]
            or len(added) != report["added_triangles"]
            or np.any(added < 0) or np.any(added > inserted)
            or not np.all(np.count_nonzero(added == inserted, axis=1) == 1)
            or np.any(added[:, 0] >= added[:, 1])
            or np.any(added[:, 0] >= added[:, 2])
            or np.any(added[:, 1] == added[:, 2])
            or len(np.unique(added, axis=0)) != len(added)):
        raise MeshError("insertion transaction has an invalid changed-cell set")
    if not np.array_equal(cell_owners[retained], old_cell_owners[old_rows]):
        raise MeshError("insertion transaction changed a retained cell owner")
    if int(owner) != -1 and (
        int(node_owners[-1]) != int(owner) or np.any(cell_owners[~retained] != int(owner))
    ):
        raise MeshError("insertion transaction lost its explicit owner")
    _positive_cells(points, added, orientation_oracle,
                    lambda phase: check("geometry"))
    removed_edges, removed_counts = _edges_and_counts(removed)
    added_edges, added_counts = _edges_and_counts(added)
    if not np.array_equal(removed_edges[removed_counts == 1],
                          added_edges[added_counts == 1]):
        raise MeshError("insertion transaction changed its cavity boundary")
    changes = {}
    for sign, edges, counts in ((-1, removed_edges, removed_counts),
                               (1, added_edges, added_counts)):
        for number, (edge, count) in enumerate(zip(edges, counts)):
            if number % 4096 == 0:
                check("incidence")
            key = tuple(map(int, edge))
            changes[key] = changes.get(key, 0) + sign * int(count)
    for number, (edge, delta) in enumerate(changes.items()):
        if number % 4096 == 0:
            check("constraints")
        incidence = len(old_index._edge_cells.get(edge, ())) + delta
        if not 0 <= incidence <= 2:
            raise MeshError("insertion transaction has invalid edge incidence")
        if incidence == 0 and (edge in protected or edge in splittable):
            raise MeshError("insertion transaction lost a constrained edge")
    check("qualified")
    from ._t3_local_validation import stage_qualified_commit
    stage_qualified_commit(points, cells, old_points, old_cells, old_index, changes)
