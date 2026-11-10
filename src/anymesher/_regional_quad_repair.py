"""Translate published quad-quality repair evidence into permanent mesh numbers."""
from .errors import MeshError
from .regional_update import _as_int, _count, _finite, _list, _mapping

__all__ = ["translate_quad_quality_repair"]


def _lookup(translate, value, path):
    try:
        return _as_int(translate(value, path), path)
    except KeyError:
        raise MeshError(f"{path} {value!r} has no permanent ID mapping; the retired ID "
                        "cannot be copied or minted") from None


# Scalar kinds: (value, ids, path) -> detached JSON value.
_SCALARS = {
    "face": lambda v, ids, p: _count(v, p),       # geometry face ID: never translated
    "count": lambda v, ids, p: _count(v, p),
    "float": lambda v, ids, p: _finite(v, p),     # signed, finite measure
    "node": lambda v, ids, p: _lookup(ids.node, v, p),
    "shell": lambda v, ids, p: _lookup(ids.shell, v, p),
}

_ROW_FACE_ELEMENT_MEASURE = ("face", "shell", "float")
_ROW_MOVE = ("face", "shell", "node", "float")
_ROW_RELOCATION = ("face", "shell", "node", "float", "float")

_CAVITY = {
    "face": "face", "triangle": "shell", "quad": "shell", "polygon": ["node"], "node": "node",
    "centroid_chart": ("float", "float"), "quality_before": "float", "quality_after": "float",
    "min_angle": "float", "cells": ["shell"],
}
_GUARD = {
    "regular_cells": "count", "protected_regular_nodes": "count",
    "blocked_triangles": [_ROW_FACE_ELEMENT_MEASURE], "cavity_repairs": [_CAVITY],
}
_REPORT = {
    "minimum_normalized_jacobian": "float",
    "split_quads": [("face", "shell", "shell", "float")],
    "normal_splits": [_ROW_FACE_ELEMENT_MEASURE],
    "relocated_for_split": [_ROW_RELOCATION],
    "relocated_quads": [_ROW_RELOCATION],
    "unresolved_quads": [_ROW_FACE_ELEMENT_MEASURE],
    "moved_nodes": [_ROW_MOVE],
    "precert_moved": [_ROW_MOVE],
    "aspect_moves": [_ROW_MOVE],
    "protected_interior_nodes": ["node"],
}
_OPTIONAL = {"regular_footprint_guard": _GUARD}


def _convert(spec, value, ids, path):
    if isinstance(spec, str):
        return _SCALARS[spec](value, ids, path)
    if isinstance(spec, dict):
        record = _mapping(value, path)
        if not all(isinstance(key, str) for key in record):
            raise MeshError(f"{path} has a non-string field name")
        if set(record) != set(spec):
            raise MeshError(f"{path} has unknown or missing fields "
                            f"{sorted(set(record) ^ set(spec), key=str)}")
        return {key: _convert(spec[key], item, ids, f"{path}.{key}") for key, item in record.items()}
    if isinstance(spec, tuple):
        row = _list(value, path)
        if len(row) != len(spec):
            raise MeshError(f"{path} row needs {len(spec)} entries, not {len(row)}")
        return [_convert(kind, item, ids, f"{path}[{i}]") for i, (kind, item) in enumerate(zip(spec, row))]
    return [_convert(spec[0], item, ids, f"{path}[{i}]") for i, item in enumerate(_list(value, path))]


def _unique(values, path):
    values = list(values)
    if len(set(values)) != len(values):
        raise MeshError(f"{path} repeats an ID")


def _check_provenance(report):
    """Reject data whose cross-references cannot be one producer run."""
    splits = report["split_quads"]
    created = [row[1] for row in splits] + [row[2] for row in splits]
    _unique(created, "quad_quality_repair.split_quads")
    _unique(report["protected_interior_nodes"], "quad_quality_repair.protected_interior_nodes")
    split_cells = {(row[0], row[1]) for row in splits}
    normal = [(row[0], row[1]) for row in report["normal_splits"]]
    _unique(normal, "quad_quality_repair.normal_splits")
    if not set(normal) <= split_cells:
        raise MeshError("quad_quality_repair.normal_splits names a cell that was not split")
    guard = report.get("regular_footprint_guard")
    if guard is None:
        return
    repairs = guard["cavity_repairs"]
    _unique([row["node"] for row in repairs], "quad_quality_repair cavity nodes")
    for row in repairs:
        where = "quad_quality_repair.regular_footprint_guard.cavity_repairs"
        _unique(row["polygon"], f"{where} polygon")
        _unique(row["cells"], f"{where} cells")
        if len(row["polygon"]) != 5 or len(row["cells"]) != 5 or row["node"] in row["polygon"]:
            raise MeshError(f"{where} is not a five-triangle fan around a new node")
        if row["cells"][:2] != [row["triangle"], row["quad"]]:
            raise MeshError(f"{where} cells do not begin with the replaced triangle and quad")


def translate_quad_quality_repair(report, ids):
    """Return a detached copy of ``report`` with node and shell IDs translated by ``ids``."""
    report = _mapping(report, "quad_quality_repair")
    schema = {**_REPORT, **{k: v for k, v in _OPTIONAL.items() if k in report}}
    out = _convert(schema, report, ids, "quad_quality_repair")
    _check_provenance(out)
    return out
