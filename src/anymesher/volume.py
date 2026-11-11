"""Immutable, admitted tet4 volume meshes; optional providers are loaded lazily.

This is a sibling of the shell Mesh, with the ANYsolid structural protocol.
Connectivity is in rows; all group membership and face ownership use external IDs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from numbers import Real
import math
import numpy as np


class VolumeMeshError(ValueError):
    """Base error for an atomic volume meshing refusal."""


class VolumeBackendUnavailable(VolumeMeshError):
    """The explicitly requested pinned provider cannot run."""


class UnsupportedVolumeGeometry(VolumeMeshError):
    """The source is outside the convex normal-prism contract."""


class InvalidVolumeMesh(VolumeMeshError):
    """Source data or provider output failed admission."""


class VolumeMeshingTimeout(VolumeMeshError):
    """The owned provider exceeded the process deadline."""


class VolumeMeshingCancelled(VolumeMeshError):
    """Cancellation was observed before publication."""


def _array(value, dtype=None):
    try:
        raw = np.asarray(value)
        if dtype in (float, np.float64) and raw.dtype.kind not in 'fiu':
            raise InvalidVolumeMesh('coordinates must contain real numerical values')
        a = np.ascontiguousarray(raw, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, InvalidVolumeMesh):
            raise
        raise InvalidVolumeMesh('invalid numerical array') from exc
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)) or value <= 0:
        raise InvalidVolumeMesh(f'{name} must be finite and positive')
    return float(value)


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise InvalidVolumeMesh('provenance keys must be strings')
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise InvalidVolumeMesh('provenance must be finite JSON-compatible data')


def _integers(value, name, shape=None):
    try:
        a = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidVolumeMesh(f'{name} must be a rectangular integer array') from exc
    if a.dtype.kind not in 'iu' or (a.dtype.kind == 'u' and a.size and a.max() > np.iinfo(np.int64).max):
        raise InvalidVolumeMesh(f'{name} must contain supported integers')
    if shape is not None and a.shape != shape:
        raise InvalidVolumeMesh(f'{name} must have shape {shape}')
    return _array(a, np.int64)


def _ids(value, count, name):
    a = _integers(np.arange(1, count + 1) if value is None else value, name, (count,))
    if len(np.unique(a)) != count:
        raise InvalidVolumeMesh(f'{name} must be unique')
    return a


def _groups(groups, ids, name):
    if groups is not None and not isinstance(groups, Mapping):
        raise InvalidVolumeMesh(f'{name} must be a mapping')
    out = {}
    for key, value in (groups or {}).items():
        if not isinstance(key, str) or not key:
            raise InvalidVolumeMesh(f'{name} names must be nonempty strings')
        a = _integers(value, name)
        if a.ndim != 1 or len(np.unique(a)) != len(a) or not np.isin(a, ids).all():
            raise InvalidVolumeMesh(f'{name}[{key}] must contain unique known external IDs')
        out[key] = a
    return MappingProxyType(out)


@dataclass(frozen=True, init=False, eq=False)
class VolumeMesh:
    """Owned immutable connected tet4 mesh with a complete outward exterior.

    Construction validates the numerical mesh. ``validate_volume_mesh`` additionally
    checks geometry ownership/coverage; generation always performs both checks.
    Supplied exterior facets must be complete, unique and outward, never a subset.
    """
    coordinates: np.ndarray
    cells: np.ndarray
    node_ids: np.ndarray
    cell_ids: np.ndarray
    boundary_facets: np.ndarray
    boundary_ids: np.ndarray
    node_groups: Mapping
    cell_groups: Mapping
    boundary_groups: Mapping
    face_owners: Mapping
    units: str
    geometry_order: int
    provenance: Mapping

    def __init__(self, coordinates, cells, node_ids=None, cell_ids=None,
                 boundary_facets=None, boundary_ids=None, node_groups=None,
                 cell_groups=None, boundary_groups=None, *, face_owners=None,
                 units='m', geometry_order=1, provenance=None, _cancel=None):
        from ._volume_validation import validate_cells
        if units != 'm' or isinstance(geometry_order, (bool, np.bool_)) or geometry_order != 1:
            raise InvalidVolumeMesh("only SI metre, geometry_order=1 tet4 meshes are supported")
        x = _array(coordinates, float)
        t = _integers(cells, 'cells')
        exterior = validate_cells(x, t, cancel=_cancel)
        if boundary_facets is not None:
            f = _integers(boundary_facets, 'boundary_facets', exterior.shape)
            if {tuple(sorted(row)) for row in f} != {tuple(sorted(row)) for row in exterior}:
                raise InvalidVolumeMesh('boundary facets must be the complete unique exterior')
            outward = {tuple(sorted(row)): row for row in exterior}
            for row in f:
                ref = outward[tuple(sorted(row))]
                if not any(np.array_equal(np.roll(row, k), ref) for k in range(3)):
                    raise InvalidVolumeMesh('boundary facets must be outward oriented')
        else:
            f = _array(exterior, np.int64)
        ni, ci, bi = (_ids(v, n, name) for v, n, name in (
            (node_ids, len(x), 'node_ids'), (cell_ids, len(t), 'cell_ids'),
            (boundary_ids, len(f), 'boundary_ids')))
        owners = dict(face_owners or {})
        if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) for k in owners) or set(owners) != set(map(int, bi)):
            raise InvalidVolumeMesh('exactly one source face owner is required per boundary external ID')
        from anygeometry.identity import EntityHandle
        if any(not isinstance(h, EntityHandle) or h.kind != 'face' for h in owners.values()):
            raise InvalidVolumeMesh('boundary owners must be source face EntityHandles')
        if len({h.model_id for h in owners.values()}) != 1:
            raise InvalidVolumeMesh('face owners must belong to one source model')
        if provenance is not None and not isinstance(provenance, Mapping):
            raise InvalidVolumeMesh('provenance must be a JSON-compatible mapping')
        values = dict(coordinates=x, cells=t, node_ids=ni, cell_ids=ci,
                      boundary_facets=f, boundary_ids=bi,
                      node_groups=_groups(node_groups, ni, 'node_groups'),
                      cell_groups=_groups(cell_groups, ci, 'cell_groups'),
                      boundary_groups=_groups(boundary_groups, bi, 'boundary_groups'),
                      face_owners=MappingProxyType(owners), units='m', geometry_order=1,
                      provenance=_freeze(provenance or {}))
        for name, value in values.items():
            object.__setattr__(self, name, value)


def validate_volume_mesh(mesh, snapshot, *, boundary_tolerance):
    """Re-admit a complete mesh against its convex source; return a frozen report."""
    from ._volume_validation import validate_source, validate_geometry
    source = validate_source(snapshot, boundary_tolerance)
    try:
        names = ('coordinates', 'cells', 'node_ids', 'cell_ids', 'boundary_facets',
                 'boundary_ids', 'node_groups', 'cell_groups', 'boundary_groups',
                 'units', 'geometry_order', 'provenance', 'face_owners')
        # Validate the whole structural protocol, including external IDs and
        # selections, rather than accepting a mutated duck object's mesh alone.
        checked = VolumeMesh(**{name: getattr(mesh, name) for name in names})
    except AttributeError as exc:
        raise InvalidVolumeMesh('object does not implement the VolumeMesh protocol') from exc
    mesh = checked
    expected_source = dict(source_model_id=source.model_id, source_revision=source.revision,
                           source_region_id=source.region_id, source_signature=source.signature)
    if any(mesh.provenance.get(key) != value for key, value in expected_source.items()):
        raise InvalidVolumeMesh('mesh provenance does not match source identity, revision and signature')
    if set(map(int, mesh.cell_groups.get(source.region_id, ()))) != set(map(int, mesh.cell_ids)):
        raise InvalidVolumeMesh('source region group must cover all external cell IDs')
    given = np.asarray(mesh.boundary_facets)
    owners, report = validate_geometry(mesh.coordinates, mesh.cells, given, source)
    if len(mesh.boundary_ids) != len(given) or len(set(map(int, mesh.boundary_ids))) != len(given):
        raise InvalidVolumeMesh('invalid boundary external IDs')
    if dict(mesh.face_owners) != {int(key): owner for key, owner in zip(mesh.boundary_ids, owners)}:
        raise InvalidVolumeMesh('source face ownership does not match geometric coverage')
    expected_groups = {name: {int(key) for key, owner in zip(mesh.boundary_ids, owners) if owner in handles}
                       for name, handles in source.groups.items()}
    if {name: set(map(int, ids)) for name, ids in mesh.boundary_groups.items()} != expected_groups:
        raise InvalidVolumeMesh('boundary groups must preserve all source selections exactly')
    return _freeze(report)


def _cancelled(cancel):
    if cancel is None:
        return False
    return bool(cancel() if callable(cancel) else cancel.is_set())


def generate_volume_mesh(snapshot, *, target_edge_length, boundary_tolerance,
                         timeout=120, cancel=None, backend='pytetwild'):
    """Generate and atomically admit a convex-prism mesh with an explicit provider.

    No retry, automatic fallback, projection or tolerance relaxation is performed.
    Cancellation accepts a callable or an object implementing ``is_set()``.
    """
    from ._volume_validation import validate_source, validate_geometry
    if backend != 'pytetwild':
        raise VolumeBackendUnavailable(f'unsupported volume backend: {backend!r}')
    for value, name in ((target_edge_length, 'target_edge_length'), (timeout, 'timeout')):
        _positive(value, name)
    source = validate_source(snapshot, boundary_tolerance)
    if _cancelled(cancel):
        raise VolumeMeshingCancelled('cancelled before provider launch')
    from ._volume_process import run_provider
    x, t, metadata = run_provider(source, float(target_edge_length), float(timeout), cancel)
    return _admit_provider_output(source, x, t, metadata, float(target_edge_length), cancel)


def _admit_provider_output(source, x, t, metadata, target_edge_length, cancel=None):
    """Common admitted-array path for generation and bounded qualification tools.

    This helper performs no provider execution and changes no capability guard.
    Qualification can inspect retained arrays through the identical production
    postchecks without enabling an unqualified native backend.
    """
    if _cancelled(cancel):
        raise VolumeMeshingCancelled('cancelled before output validation')
    from ._volume_validation import validate_cells, validate_geometry
    # Provider orientation convention is canonicalized explicitly; never repair a
    # collapsed cell, position or topology. Keep every raw row for admission.
    x = _array(x, float)
    t = _integers(t, 'provider cells').copy()
    if x.ndim != 2 or x.shape[1:] != (3,) or t.ndim != 2 or t.shape[1:] != (4,):
        raise InvalidVolumeMesh('provider arrays must be (N,3) and (M,4)')
    if not len(t) or t.min() < 0 or t.max() >= len(x):
        raise InvalidVolumeMesh('provider cells reference unknown rows or are empty')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        d = x[t[:, 1:]] - x[t[:, :1]]
        det = np.linalg.det(d / np.max(np.abs(d), axis=(1, 2))[:, None, None])
    reverse = det < 0
    t[reverse, 1], t[reverse, 2] = t[reverse, 2].copy(), t[reverse, 1].copy()
    # Canonical row order and recoverable raw-provider row maps. This only
    # permutes rows/connectivity: it never merges, drops or moves a vertex/cell.
    node_order = np.lexsort((x[:, 2], x[:, 1], x[:, 0]))
    raw_node_to_row = np.empty(len(x), dtype=np.int64)
    raw_node_to_row[node_order] = np.arange(len(x))
    x = x[node_order]
    t = np.sort(raw_node_to_row[t], axis=1)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        d = x[t[:, 1:]] - x[t[:, :1]]
        canonical_reverse = np.linalg.det(d / np.max(np.abs(d), axis=(1, 2))[:, None, None]) < 0
    t[canonical_reverse, 2], t[canonical_reverse, 3] = t[canonical_reverse, 3].copy(), t[canonical_reverse, 2].copy()
    cell_order = np.lexsort(tuple(t[:, k] for k in range(3, -1, -1)))
    raw_cell_to_row = np.empty(len(t), dtype=np.int64)
    raw_cell_to_row[cell_order] = np.arange(len(t))
    t = t[cell_order]
    exterior = validate_cells(x, t, cancel=cancel)
    owners, report = validate_geometry(x, t, exterior, source, cancel=cancel)
    ids = np.arange(1, len(exterior) + 1)
    groups = {name: ids[[h in handles for h in owners]] for name, handles in source.groups.items()}
    metadata.update(source_model_id=source.model_id, source_revision=source.revision,
                    source_region_id=source.region_id, source_signature=source.signature,
                    source_provenance=source.provenance,
                    contract='solid_volume_handoff_v1', boundary_tolerance=source.delta,
                    target_edge_length=float(target_edge_length), validation=report,
                    orientation_swaps=int(np.count_nonzero(reverse)),
                    canonicalization='lexicographic coordinates and positive tet4 rows',
                    raw_node_row_to_row=raw_node_to_row.tolist(),
                    raw_cell_row_to_row=raw_cell_to_row.tolist())
    if _cancelled(cancel):
        raise VolumeMeshingCancelled('cancelled before publication')
    # Constructor rechecks independent immutable storage. No object escapes until
    # every numerical/geometry/group check above succeeds.
    mesh = VolumeMesh(x, t, boundary_facets=exterior, boundary_ids=ids,
                      cell_groups={source.region_id: np.arange(1, len(t) + 1)},
                      boundary_groups=groups, face_owners=dict(zip(map(int, ids), owners)),
                      provenance=metadata, _cancel=cancel)
    if _cancelled(cancel):
        raise VolumeMeshingCancelled('cancelled before publication')
    return mesh
