"""Exact changes of labels on a live source-published shell mesh."""

from copy import copy, deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from weakref import ref

from .errors import MeshError
from . import material_source_transfer as transfer
from . import regional_update as regional
from ._material_region_binding import boundary_registry_pin


_issued = {}
_numbered = {}


def _remember(table, mesh, state):
    key = id(mesh)
    def cleanup(reference):
        if key in table and table[key][0] is reference:
            del table[key]
    table[key] = (ref(mesh, cleanup), state)


def _get(table, mesh):
    row = table.get(id(mesh))
    return row[1] if row and row[0]() is mesh else None


def _report_digest(mesh):
    from .serialize import _json_value
    # Caller timings and counts are finalized after owner publication. They
    # are independent of geometry authority and checked by the report codec.
    reports = {k: v for k,v in mesh.hybrid_diagnostics.items() if k not in ('runtime', 'automatic_selection')}
    payload = _json_value([mesh.structural_preparation, reports], 'source publication reports')
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).digest()


def remember_source_publication(mesh):
    """Called only at the final successful owner-publication boundary."""
    _remember(_issued, mesh, (transfer._IDENTITY_PUBLICATIONS[id(mesh)], _report_digest(mesh)))


@dataclass(frozen=True)
class _NumberingProof:
    original: object
    elements: dict
    registry: object
    stations: tuple
    cells: tuple
    reports: bytes


def _original(mesh, source):
    proof = _get(_numbered, mesh)
    if proof is not None:
        if (mesh.boundary_registry is not proof.registry or boundary_registry_pin(proof.registry) != proof.stations
                or transfer._publication_pin(mesh) != proof.cells or _report_digest(mesh) != proof.reports):
            raise MeshError('numbered source publication cells, stations or reports changed')
    original = proof.original if proof is not None else mesh
    issued = _get(_issued, original)
    if issued is None or issued[0] is not transfer._IDENTITY_PUBLICATIONS.get(id(original)):
        raise MeshError('permanent numbering requires the original live material source authority')
    entry, reports = issued
    transfer._validate_identity_source_publication(source, entry[2], original, original.boundary_registry)
    if reports != _report_digest(original):
        raise MeshError('original source publication reports changed')
    return original, proof


def validate_numbered_source_mesh(mesh, *, source_geometry):
    """Validate the live source-cell evidence and its exact label transport."""
    _original(mesh, source_geometry)


def numbered_reference_uv(source, mesh):
    proof = _get(_numbered, mesh)
    if proof is None:
        return None
    original, _ = _original(mesh, source)
    return {proof.elements[e]: uv for e, uv in transfer._validated_curvilinear_reference_uv(source, original).items()}


def _trace_provenance(record, nm, em):
    row = deepcopy(record)
    row['vertices'] = {k: nm[n] for k,n in row['vertices'].items()}
    for trace in row['traces'].values():
        if set(trace) != {'prepared_model_id','prepared_edge_id','source_face_ids','stations','elements','offset_nodes'}:
            raise MeshError('unrecognized prepared boundary trace record')
        trace['stations'] = [(t, nm[n], xyz) for t,n,xyz in trace['stations']]
        trace['elements'] = [em[e] for e in trace['elements']]
        trace['offset_nodes'] = [nm[n] for n in trace['offset_nodes']]
    return row


_MATERIAL_FACE_FIELDS = {'native_v2','material_region','analytic_physical_quality',
                         'material_region_final_physical_quality','detached_material_evaluations',
                         'parametric_tensor', 'parametric_tensor_quality', 'parametric_tensor_refusal'}


def _face_reports(reports, nm):
    # SurfaceMeshOptions diagnostics and physical quality are local MeshCore
    # row indices. Only native-v2 shared station receipts use global node IDs.
    reports = deepcopy(reports)
    for row in reports.values():
        unexpected = set(row) - (regional._BACKEND_REQUIRED | regional._BACKEND_OPTIONAL |
                                 set(regional._NATIVE_FACE_REPORT) | _MATERIAL_FACE_FIELDS)
        if unexpected:
            raise MeshError(f'unknown source face generation report: {sorted(unexpected)}')
        translated = set()
        for native in (row.get('native_v2', {}), row.get('native_diagnostics', {}).get('native_v2', {})):
            for station in native.get('shared_nodes', ()):
                if id(station) not in translated:
                    station['node_id'] = nm[station['node_id']]
                    translated.add(id(station))
        if row.get('working_face_diagnostics'):
            raise MeshError('nested working-face source diagnostics require explicit label transport')
    return reports


def remap_source_mesh_ids(mesh, node_ids, element_ids, *, source_geometry):
    """Rebind genuine, unchanged source cells under a total bijection of labels.

    The original generator and geometry remain the proof for each physical cell.
    Transported artifacts cannot issue new source authority. A new cell or a
    change of connectivity requires its own generator's publication.
    """
    original, prior = _original(mesh, source_geometry)
    nodes, elements = regional._validate_structure(mesh, 'source mesh', allow_zero=True)
    nm = regional._check_map('node_ids', node_ids, nodes)
    em = regional._check_map('element_ids', element_ids, elements)
    if all(k == v for k,v in nm.items()) and all(k == v for k,v in em.items()):
        return mesh
    ids = regional._Ids(nodes, elements, set(mesh.tris), nm, em, quads=mesh.quads)
    header = copy(mesh)
    header.structural_preparation = dict(mesh.structural_preparation)
    header.hybrid_diagnostics = dict(mesh.hybrid_diagnostics)
    provenance = header.structural_preparation.pop('prepared_boundary_provenance', None)
    closure = header.structural_preparation.pop('closure_source_transfer', None)
    faces = header.hybrid_diagnostics.pop('triangulation_backend_by_face', None)
    error = header.hybrid_diagnostics.pop('geometry_approximation_error', None)
    prep, diagnostics, _ = regional._process_metadata(header, 'source mesh', ids)
    if provenance is not None:
        prep['prepared_boundary_provenance'] = _trace_provenance(provenance, nm, em)
    if closure is not None:
        prep['closure_source_transfer'] = deepcopy(closure)
    if faces is not None:
        diagnostics['triangulation_backend_by_face'] = _face_reports(faces, nm)
    if error is not None:
        diagnostics['geometry_approximation_error'] = error
    result = regional._translate(mesh, nm, em, prep, diagnostics)
    from .boundary import GlobalEdgeBoundaryRegistry
    registry = GlobalEdgeBoundaryRegistry(mesh.boundary_registry.view)
    for station in mesh.boundary_registry.entries():
        for owner in station.owners or (None,):
            registry.register(station.key.edge_id, station.key.parameter, station.point,
                              node_id=None if station.node_id is None else nm[station.node_id], owner=owner)
    result.boundary_registry = registry
    mapping = em if prior is None else {k: em[v] for k,v in prior.elements.items()}
    _original(mesh, source_geometry)
    _remember(_numbered, result, _NumberingProof(original, mapping, registry, boundary_registry_pin(registry),
                                                 transfer._publication_pin(result), _report_digest(result)))
    return result
