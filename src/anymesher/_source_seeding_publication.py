"""Transport solved edge-count variables through authenticated source partitions.

The caller first authenticates owner lineage and literal boundary publication.
This module grants no geometry or meshing admission.
"""
from collections import Counter
from copy import copy
from fractions import Fraction
from math import isfinite
from numbers import Integral

from .errors import MeshError


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
        from .refinement import SizeField
        field = original.size_field
        if (type(field) is not SizeField or any(getattr(field, name, None)
                for name in ('zones', 'overrides', '_sources', 'face_sizes', 'edge_sizes', 'beam_sizes'))):
            raise MeshError('source seeding cannot remap prepared refinement references')
        target = field.target_size
        if isinstance(target, bool) or not isinstance(target, (int,float)) or not isfinite(target) or target <= 0:
            raise MeshError('source seeding size field target is invalid')
        result.size_field = SizeField(source, target)
    return result
