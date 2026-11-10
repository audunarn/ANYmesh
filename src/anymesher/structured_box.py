"""Explicit structured Tet4 lattice for one rectangular-box solid region.

This is a provider-free sibling of ``generate_volume_mesh``. It does not use,
enable or alter the native ``pytetwild`` capability gate. The lattice is built
by trilinear interpolation of the snapshot's eight corners (so corners are
bit-exact), every hexahedral lattice cell is split into six tetrahedra around
its main diagonal (Kuhn subdivision, conforming across all cell faces) and the
result is published only through the same admitted-array path as the provider
route: canonical rows, positive Tet4s, ``validate_cells`` and
``validate_geometry`` ownership/coverage, source groups and identities.
"""
from __future__ import annotations

from itertools import permutations
from numbers import Integral
import math

import numpy as np

from .volume import (InvalidVolumeMesh, UnsupportedVolumeGeometry,
                     VolumeMeshingCancelled, _admit_provider_output,
                     _cancelled, _positive)

__all__ = ['generate_structured_box_mesh', 'MAX_STRUCTURED_BOX_CELLS',
           'MAX_STRUCTURED_BOX_AXIS_DIVISIONS']

#: Resource guards, checked with Python integers before any array is allocated.
#: Admission is pure-Python/NumPy exhaustive validation; these are allocation
#: guards, not scalability claims.
MAX_STRUCTURED_BOX_CELLS = 120_000
MAX_STRUCTURED_BOX_AXIS_DIVISIONS = 4_096

_RELATIVE_BOX_TOLERANCE = 1e-9
_PERMUTATIONS = tuple(permutations(range(3)))


def _parity(p):
    inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if p[i] > p[j])
    return -1 if inversions % 2 else 1


def _poll(cancel, where):
    if _cancelled(cancel):
        raise VolumeMeshingCancelled(f'cancelled {where}')


def _layers(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise InvalidVolumeMesh('layers must be a positive integer')
    return int(value)


def _same_cycle(loop, wanted):
    loop = tuple(int(k) for k in loop)
    return len(loop) == len(wanted) and any(loop == wanted[i:] + wanted[:i] for i in range(len(wanted)))


def _box_frame(source):
    """Return the eight corners and edge vectors, or refuse unsupported topology."""
    x = source.coordinates
    if len(x) != 8 or len(source.loops) != 6:
        raise UnsupportedVolumeGeometry('structured box meshing requires a four-sided prism profile')
    base = [(3, 2, 1, 0), (4, 5, 6, 7)] + [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4) for i in range(4)]
    forward = all(_same_cycle(l, w) for l, w in zip(source.loops, base))
    reverse = all(_same_cycle(l, tuple(reversed(w))) for l, w in zip(source.loops, base))
    if not (forward or reverse):
        raise UnsupportedVolumeGeometry('snapshot face roles do not follow the start/end/side_i prism layout')
    a, b, c = x[1] - x[0], x[3] - x[0], x[4] - x[0]
    la, lb, lc = (float(np.linalg.norm(v)) for v in (a, b, c))
    scale = source.diagonal * _RELATIVE_BOX_TOLERANCE
    if min(la, lb, lc) <= 0 or not all(map(math.isfinite, (la, lb, lc))):
        raise UnsupportedVolumeGeometry('degenerate box edge')
    if (np.linalg.norm(x[2] - (x[1] + x[3] - x[0])) > scale
            or any(np.linalg.norm(x[k + 4] - x[k] - c) > scale for k in range(4))):
        raise UnsupportedVolumeGeometry('source is not a parallelepiped box')
    for u, v, lu, lv in ((a, b, la, lb), (a, c, la, lc), (b, c, lb, lc)):
        if abs(float(u @ v)) > _RELATIVE_BOX_TOLERANCE * lu * lv:
            raise UnsupportedVolumeGeometry('source is not an orthogonal rectangular box')
    handed = 1 if float(a @ np.cross(b, c)) > 0 else -1
    return x, (la, lb, lc), handed


def _divisions(length, target):
    ratio = length / target
    if not math.isfinite(ratio) or ratio > MAX_STRUCTURED_BOX_AXIS_DIVISIONS * 2:
        raise InvalidVolumeMesh('structured box lattice exceeds the axis division resource limit')
    n = max(1, math.ceil(ratio * (1 - 1e-12)))
    if n > MAX_STRUCTURED_BOX_AXIS_DIVISIONS:
        raise InvalidVolumeMesh('structured box lattice exceeds the axis division resource limit')
    return n


def generate_structured_box_mesh(snapshot, *, target_edge_length, layers,
                                 boundary_tolerance, cancel=None):
    """Mesh one orthogonal rectangular box with an explicit conforming Tet4 lattice.

    ``target_edge_length`` (metres) independently sizes the two in-plane profile
    axes: each gets ``ceil(length / target_edge_length)`` divisions, so no axis
    edge exceeds the target. ``layers`` is the exact positive number of cell layers
    through the extrusion thickness. The two profile axes are the
    ``start`` face edges ``x1-x0`` and ``x3-x0``; the thickness axis is ``x4-x0``.

    Refusals: ``InvalidVolumeMesh`` for invalid sizes, layers, tolerance, malformed
    snapshots or an impossible lattice (resource guard);
    ``UnsupportedVolumeGeometry`` for any source that is not an orthogonal box;
    ``VolumeMeshingCancelled`` when cancellation is observed during generation,
    admission or before publication. Nothing is published on refusal.
    """
    from ._volume_validation import validate_source
    _positive(target_edge_length, 'target_edge_length')
    nz = _layers(layers)
    source = validate_source(snapshot, boundary_tolerance)
    _poll(cancel, 'before structured box generation')
    x, lengths, handed = _box_frame(source)
    target = float(target_edge_length)
    nx, ny = _divisions(lengths[0], target), _divisions(lengths[1], target)
    if nz > MAX_STRUCTURED_BOX_AXIS_DIVISIONS or 6 * nx * ny * nz > MAX_STRUCTURED_BOX_CELLS:
        raise InvalidVolumeMesh(
            f'structured box lattice of {nx}x{ny}x{nz} hexahedra exceeds the cell resource limit')

    s = np.arange(nx + 1, dtype=np.float64) / nx
    t = np.arange(ny + 1, dtype=np.float64) / ny
    sw, tw = s[:, None], t[None, :]
    plane = ((1 - sw) * (1 - tw), sw * (1 - tw), (1 - sw) * tw, sw * tw)
    sx, sy = nx + 1, ny + 1
    nodes = np.empty((nz + 1, sy, sx, 3))
    for k in range(nz + 1):
        _poll(cancel, 'during structured box node generation')
        u = k / nz
        low = sum(w[..., None] * x[i] for w, i in zip(plane, (0, 1, 3, 2)))
        high = sum(w[..., None] * x[i] for w, i in zip(plane, (4, 5, 7, 6)))
        # Weights at the end layers are exactly 0/1, so snapshot corners are bit-exact.
        nodes[k] = ((1 - u) * low + u * high).transpose(1, 0, 2)
    coordinates = nodes.reshape(-1, 3)

    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    offsets = (1, sx, sx * sy)
    parts = []
    for k in range(nz):
        _poll(cancel, 'during structured box cell generation')
        v0 = ((k * sy + j) * sx + i).ravel().astype(np.int64)
        v7 = v0 + sum(offsets)
        for p in _PERMUTATIONS:
            o1 = offsets[p[0]]
            o2 = o1 + offsets[p[1]]
            cell = np.stack((v0, v0 + o1, v0 + o2, v7), axis=1)
            if _parity(p) * handed < 0:
                cell[:, [2, 3]] = cell[:, [3, 2]]
            parts.append(cell)
    cells = np.concatenate(parts)
    _poll(cancel, 'before structured box admission')
    metadata = dict(
        backend='structured_box', provider='none; explicit lattice',
        structured_box=dict(divisions=[nx, ny, nz], layers=nz,
                            edge_lengths=[lengths[0] / nx, lengths[1] / ny, lengths[2] / nz],
                            box_lengths=list(lengths),
                            subdivision='six conforming tetrahedra per hexahedron around the main diagonal',
                            max_cells=MAX_STRUCTURED_BOX_CELLS))
    return _admit_provider_output(source, coordinates, cells, metadata, target, cancel)
