import json
from copy import deepcopy

import pytest

from anymesher._regional_quad_repair import translate_quad_quality_repair
from anymesher.errors import MeshError
from anymesher.regional_update import _Ids

NODES = {1, 2, 3, 4, 5, 6, 7, 8, 9}
TRIS = {10, 11, 12, 13, 14, 15, 16}
QUADS = {20}
NODE_MAP = {n: 100 + 7 * n for n in NODES}            # sparse and non-contiguous
ELEMENT_MAP = {e: 1000 + 13 * e for e in TRIS | QUADS}


def make_ids(node_map=NODE_MAP, element_map=ELEMENT_MAP):
    return _Ids(set(NODES), TRIS | QUADS, set(TRIS), node_map, element_map, quads=QUADS)


def empty_report(guard=False):
    report = {
        "minimum_normalized_jacobian": 0.1, "split_quads": [], "normal_splits": [],
        "relocated_for_split": [], "relocated_quads": [], "unresolved_quads": [],
        "moved_nodes": [], "precert_moved": [], "aspect_moves": [], "protected_interior_nodes": (),
    }
    if guard:
        report["regular_footprint_guard"] = {
            "regular_cells": 4, "protected_regular_nodes": 9, "blocked_triangles": [], "cavity_repairs": []}
    return report


def full_report():
    report = empty_report(guard=True)
    report.update({
        "split_quads": [(3, 10, 11, -0.25), (3, 12, 13, 0.01)],
        "normal_splits": [(3, 12, 31.5)],
        "relocated_for_split": [(3, 10, 2, 0.125, 0.3)],
        "relocated_quads": [(5, 20, 4, 0.5, -0.5)],
        "moved_nodes": [(3, 11, 5, 0.01)],
        "precert_moved": [(3, 13, 6, 0.02)],
        "aspect_moves": [(5, 20, 7, 0.03)],
        "protected_interior_nodes": (9, 1),
    })
    report["regular_footprint_guard"]["cavity_repairs"] = [{
        "face": 5, "triangle": 14, "quad": 15, "polygon": [1, 2, 3, 4, 5], "node": 8,
        "centroid_chart": [0.5, -0.25], "quality_before": 0.01, "quality_after": 0.2,
        "min_angle": 20.0, "cells": [14, 15, 16, 12, 13]}]
    return report


def test_empty_production_report_roundtrips_for_identity_and_sparse_maps():
    for guard in (False, True):
        report = empty_report(guard)
        for ids in (_Ids(set(NODES), TRIS | QUADS, set(TRIS), quads=QUADS), make_ids()):
            out = translate_quad_quality_repair(report, ids)
            assert out == json.loads(json.dumps(out))
            assert out["protected_interior_nodes"] == []
            assert out["minimum_normalized_jacobian"] == 0.1
            assert ("regular_footprint_guard" in out) is guard


def test_nonempty_split_and_move_report_is_translated_and_detached():
    report = full_report()
    before = deepcopy(report)
    out = translate_quad_quality_repair(report, make_ids())
    n, e = NODE_MAP, ELEMENT_MAP
    assert report == before
    assert out["split_quads"] == [[3, e[10], e[11], -0.25], [3, e[12], e[13], 0.01]]
    assert out["normal_splits"] == [[3, e[12], 31.5]]
    assert out["relocated_for_split"] == [[3, e[10], n[2], 0.125, 0.3]]
    assert out["relocated_quads"] == [[5, e[20], n[4], 0.5, -0.5]]
    assert out["moved_nodes"] == [[3, e[11], n[5], 0.01]]
    assert out["precert_moved"] == [[3, e[13], n[6], 0.02]]
    assert out["aspect_moves"] == [[5, e[20], n[7], 0.03]]
    assert out["protected_interior_nodes"] == [n[9], n[1]]
    cavity = out["regular_footprint_guard"]["cavity_repairs"][0]
    assert cavity["face"] == 5 and cavity["node"] == n[8]
    assert cavity["polygon"] == [n[i] for i in (1, 2, 3, 4, 5)]
    assert cavity["triangle"] == e[14] and cavity["quad"] == e[15]
    assert cavity["cells"] == [e[c] for c in (14, 15, 16, 12, 13)]
    assert cavity["centroid_chart"] == [0.5, -0.25] and cavity["min_angle"] == 20.0
    assert out["regular_footprint_guard"]["regular_cells"] == 4
    assert out == json.loads(json.dumps(out))
    out["split_quads"][0][0] = 99
    out["regular_footprint_guard"]["cavity_repairs"][0]["polygon"].append(0)
    assert report == before


def test_json_roundtripped_input_gives_same_result():
    report = full_report()
    assert (translate_quad_quality_repair(json.loads(json.dumps(report)), make_ids())
            == translate_quad_quality_repair(report, make_ids()))


def corrupt(mutate):
    report = full_report()
    mutate(report)
    return report


def _cavity(report):
    return report["regular_footprint_guard"]["cavity_repairs"][0]


BAD_REPORTS = {
    "unknown_key": lambda r: r.update(surprise=[]),
    "missing_key": lambda r: r.pop("aspect_moves"),
    "unknown_guard_key": lambda r: r["regular_footprint_guard"].update(extra=0),
    "unknown_cavity_key": lambda r: _cavity(r).update(extra=0),
    "foreign_node": lambda r: r["moved_nodes"].append((3, 11, 99, 0.0)),
    "foreign_shell": lambda r: r["moved_nodes"].append((3, 99, 5, 0.0)),
    "foreign_protected": lambda r: r.update(protected_interior_nodes=(99,)),
    "foreign_cavity_node": lambda r: _cavity(r).update(node=99),
    "foreign_cavity_cell": lambda r: _cavity(r)["cells"].__setitem__(4, 99),
    "negative_count": lambda r: r["regular_footprint_guard"].update(regular_cells=-1),
    "bool_count": lambda r: r["regular_footprint_guard"].update(protected_regular_nodes=True),
    "float_face": lambda r: r["split_quads"].__setitem__(0, (3.0, 10, 11, 0.0)),
    "nan_measure": lambda r: r["aspect_moves"].__setitem__(0, (5, 20, 7, float("nan"))),
    "inf_measure": lambda r: r.update(minimum_normalized_jacobian=float("inf")),
    "short_row": lambda r: r["precert_moved"].__setitem__(0, (3, 13, 6)),
    "row_not_list": lambda r: r.update(moved_nodes="abc"),
    "duplicate_protected": lambda r: r.update(protected_interior_nodes=(1, 1)),
    "split_reuses_cell": lambda r: r["split_quads"].append((3, 10, 14, 0.0)),
    "normal_split_not_split": lambda r: r["normal_splits"].append((3, 14, 1.0)),
    "cavity_node_in_polygon": lambda r: _cavity(r).update(node=1),
    "cavity_short_polygon": lambda r: _cavity(r).update(polygon=[1, 2, 3, 4]),
    "cavity_cells_disagree": lambda r: _cavity(r).update(triangle=16),
    "cavity_bad_centroid": lambda r: _cavity(r).update(centroid_chart=[0.5]),
}


@pytest.mark.parametrize("name", sorted(BAD_REPORTS))
def test_corrupt_reports_are_refused(name):
    with pytest.raises(MeshError):
        translate_quad_quality_repair(corrupt(BAD_REPORTS[name]), make_ids())


def test_report_must_be_a_mapping():
    with pytest.raises(MeshError):
        translate_quad_quality_repair([], make_ids())


def test_retired_ids_without_mapping_fail_instead_of_being_copied():
    report = full_report()
    node_map = {k: v for k, v in NODE_MAP.items() if k != 5}      # node 5 is published but unmapped
    with pytest.raises(MeshError, match="no permanent ID mapping"):
        translate_quad_quality_repair(report, make_ids(node_map=node_map))
    element_map = {k: v for k, v in ELEMENT_MAP.items() if k != 13}
    with pytest.raises(MeshError, match="no permanent ID mapping"):
        translate_quad_quality_repair(report, make_ids(element_map=element_map))


def test_non_injective_transform_is_refused():
    node_map = {**NODE_MAP, 2: NODE_MAP[1]}
    with pytest.raises(MeshError):
        translate_quad_quality_repair(full_report(), make_ids(node_map=node_map))
    element_map = {**ELEMENT_MAP, 11: ELEMENT_MAP[10]}
    with pytest.raises(MeshError):
        translate_quad_quality_repair(full_report(), make_ids(element_map=element_map))
