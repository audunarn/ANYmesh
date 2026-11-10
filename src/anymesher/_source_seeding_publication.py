"""Transport solved edge-count variables through authenticated source partitions.

The caller first authenticates owner lineage and literal boundary publication.
This module grants no geometry or meshing admission.
"""
from collections import Counter
from copy import copy
from fractions import Fraction
from math import isfinite
from numbers import Integral

import numpy as np

from .errors import MeshError


def _source_size_field(source, field):
    from .refinement import Refinement, SizeField
    expected = {'zones','overrides','_sources','face_sizes','edge_sizes','beam_sizes','target_size'}
    if (type(field) is not SizeField or set(vars(field)) != expected
            or any(getattr(field, k) for k in ('overrides','face_sizes','edge_sizes','beam_sizes'))):
        raise MeshError('source seeding cannot remap prepared refinement references')
    if type(field.target_size) is not float or not isfinite(field.target_size) or field.target_size <= 0:
        raise MeshError('source seeding size field target is invalid')
    if type(field.zones) is not tuple or len(field.zones) != len(field._sources):
        raise MeshError('source seeding point field differs from its prepared coordinates')
    points = []
    for zone, cached in zip(field.zones, field._sources):
        if (type(zone) is not Refinement or zone.mode != 'refine' or zone.ref is not None
                or zone.center is None or type(zone.name) is not str):
            raise MeshError('source seeding cannot remap prepared refinement references')
        if type(cached) is not np.ndarray or not np.array_equal(cached, [zone.center]):
            raise MeshError('source seeding point field differs from its prepared coordinates')
        points.append(Refinement(center=zone.center, size=zone.size, radius=zone.radius,
                                 growth=zone.growth, name=zone.name))
    return SizeField(source, field.target_size, points)


def source_size_field_metadata(source, field):
    """Portable sizing request from a validated, source-projected literal field."""
    if field is None:
        return None
    result = {'kind': 'uniform', 'target_size': field.target_size, 'source_model_id': str(source.model_id)}
    if field.zones:
        result.update(kind='point_refinements', zones=[
            {'center': list(z.center), 'size': z.size, 'radius': z.radius, 'growth': z.growth,
             'name': z.name, 'mode': z.mode} for z in field.zones])
    return result


def project_source_seeding(source, mesh, receipt, edge_maps, source_nodes):
    original = mesh.seeding
    if original is None:
        return None
    mapped = set(edge_maps)
    known = mapped | {edge.id for edge, _owners in receipt.source_trace_ownership}
    divisions, classes = original.divisions, original.classes

    def integer(value, *, positive=False):
        if isinstance(value, bool) or not isinstance(value, Integral) or (positive and value <= 0):
            raise MeshError('source seeding requires integral positive counts and valid class references')
        return int(value)

    if any(integer(edge) not in known for edge in set(divisions) | set(classes)):
        raise MeshError('prepared seeding lacks owner edge or trace ancestry')
    if (classes and set(divisions) != set(classes)) or not mapped <= set(divisions):
        raise MeshError('source seeding lacks complete division and class assignments')
    # Counts are solver variables; owner chord refinement may add stations without
    # changing these advisory counts. Authenticate chains independently below.
    counts, class_counts = {}, {}
    for edge, value in divisions.items():
        count = integer(value, positive=True)
        if classes:
            label = integer(classes[edge])
            if label not in classes or integer(classes[label]) != label:
                raise MeshError('source seeding class lacks its canonical prepared member')
            if label in class_counts and class_counts[label] != count:
                raise MeshError('source seeding class members have conflicting division counts')
            class_counts[label] = count
        counts[edge] = count
        chain = list(mesh.nodes_of_edge.get(edge, ()))
        stations = mesh.boundary_registry.entries(edge)
        if (len(chain) < 2 or len(stations) != len(chain)
                or [row.node_id for row in stations] != chain
                or any(node not in mesh.nodes for node in chain)
                or not stations or stations[0].key.parameter != 0 or stations[-1].key.parameter != 1):
            raise MeshError('source seeding lacks a consistent literal prepared station chain')
    parents = {}
    for edge, (parent, lo, hi) in edge_maps.items():
        lo, hi = Fraction(lo), Fraction(hi)
        if parent.id not in source.edges or parent.model_id != source.model_id or lo == hi:
            raise MeshError('source seeding edge partition lacks source ancestry')
        chain = list(mesh.nodes_of_edge[edge])
        parents.setdefault(parent.id, []).append((min(lo, hi), max(lo, hi), edge,
                                                   chain if lo < hi else chain[::-1]))
    projected, expressions = {}, {}
    for parent, spans in sorted(parents.items()):
        spans.sort(key=lambda span: span[0])
        if (spans[0][0] != 0 or spans[-1][1] != 1
                or any(a[1] != b[0] for a,b in zip(spans,spans[1:]))):
            raise MeshError('source seeding edge partition is incomplete or overlapping')
        chain = list(spans[0][3])
        for _lo, _hi, _edge, next_chain in spans[1:]:
            if chain[-1] != next_chain[0]:
                raise MeshError('source seeding partition stations have distinct joint identities')
            chain.extend(next_chain[1:])
        total = sum(counts[edge] for _lo,_hi,edge,_chain in spans)
        if chain != list(source_nodes.get(parent, ())):
            raise MeshError('source seeding differs from authenticated source station chain')
        projected[parent] = total
        if classes:
            expressions[parent] = tuple(sorted(Counter(int(classes[edge])
                for _lo,_hi,edge,_chain in spans).items()))
    representatives, projected_classes = {}, {}
    for parent in sorted(expressions):
        signature = expressions[parent]
        projected_classes[parent] = representatives.setdefault(signature, parent)
    result = copy(original)
    result.divisions, result.classes = projected, projected_classes
    if original.size_field is not None:
        result.size_field = _source_size_field(source, original.size_field)
    return result
