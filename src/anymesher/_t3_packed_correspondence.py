"""Lossless integer comparison for ordered, bounded T3 connectivity.

This is an exact representation of integer row keys, never a coordinate
quantization or hash. Unsupported domains retain the independent reference.
"""
from __future__ import annotations

import numpy as np


def _encode(rows, bits):
    values = rows.view(np.uint64)
    result = values[:, 0] << (2 * bits)
    result |= values[:, 1] << bits
    result |= values[:, 2]
    return result


def packed_source_rows(before, after):
    """Return exact first-source matches, or None for the general path."""
    for rows in (before, after):
        if (not isinstance(rows, np.ndarray)
                or rows.dtype != np.dtype(np.int64)
                or rows.ndim != 2 or rows.shape[1] != 3):
            return None
        if np.any(rows < 0):
            return None
    maximum = max(int(before.max(initial=0)), int(after.max(initial=0)))
    bits = max(1, maximum.bit_length())
    if bits > 21:
        return None
    old_keys = _encode(before, bits)
    if np.any(old_keys[1:] < old_keys[:-1]):
        return None
    keys = _encode(after, bits)
    positions = np.searchsorted(old_keys, keys, side="left")
    matched = positions < len(old_keys)
    candidates = np.flatnonzero(matched)
    matched[candidates] = old_keys[positions[candidates]] == keys[candidates]
    result = np.full(len(after), -1, dtype=np.int64)
    result[matched] = positions[matched]
    return result
