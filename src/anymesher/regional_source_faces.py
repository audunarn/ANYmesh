"""Reissue disconnected source-face cells through fresh ordinary generation proofs."""
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
import json

import numpy as np

from .core import MeshCore
from .errors import MeshError
from ._row_provenance import OutputRowProvenance

_ACTIVE = ContextVar("regional_source_face_candidates", default=None)


def _digest(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _vertex_faces(model):
    incidence = {}
    for face in model.faces.values():
        for loop in (face.loop, *face.holes):
            for use in loop:
                edge = model.edges[use.edge]
                for vertex in (edge.start, edge.end):
                    incidence.setdefault(vertex, set()).add(face.id)
    return incidence


def _native_definitions(model):
    from anygeometry.native_support_snapshots import capture_native_supports
    return {row[0]: row for row in capture_native_supports(model)}


def _ordinary_face(model, face_id, vertex_faces):
    from anygeometry.nurbs import NurbsSurface
    from ._material_region_binding import polynomial_tensor_chart
    face = model.faces[face_id]
    if (face.holes or face.parameterization is not None
            or len(face.loop) != 4 or len({use.edge for use in face.loop}) != 4):
        return False
    if type(face.surface) is not NurbsSurface and polynomial_tensor_chart(model, face_id) is None:
        return False
    vertices = {model.edges[u.edge].start for u in face.loop} | {model.edges[u.edge].end for u in face.loop}
    return all(vertex_faces[v] == {face_id} for v in vertices)


def _face_signature(model, face_id, native):
    """Pin the complete retained rectangle, its occurrences and numerical policy."""
    from anygeometry.definition_binding import definition_checksum
    face = model.faces[face_id]
    edges = tuple(model.edges[u.edge] for u in face.loop)
    vertex_ids = {v for e in edges for v in (e.start, *getattr(e.curve,'control_vertices',()), e.end)}
    vertices = tuple(model.vertices[v] for v in sorted(vertex_ids))
    uses = tuple(model.face_uses[u] for u in sorted(model._face_structural_uses.get(face_id, ())))
    sheets = tuple(model.sheets[s] for s in sorted({use.sheet_id for use in uses}))
    return definition_checksum((model.model_id, model.tolerance, face, edges, vertices, uses, sheets, native))


def reusable_face_for_request(record, strategy, recombine):
    """Check the proposal's element and grid capabilities for a planned unchanged face."""
    data = record.get('data', {})
    if strategy not in ('native', 'mapped', 'auto', None):
        return False
    if strategy == 'mapped' and not data.get('grid'):
        return False
    cells = data.get('cells', ())
    return bool(cells) and (recombine or all(len(c)==3 for c in cells))


def capture_reusable_source_faces(source, mesh):
    """Export proposals from a live validated closure publication, in original source coordinates."""
    from . import material_source_transfer as transfer
    from .regional_source_numbering import _original
    vertex_faces = _vertex_faces(source)
    faces = [f for f in source.faces if _ordinary_face(source, f, vertex_faces)]
    if not faces:
        return {}
    original, _ = _original(mesh, source)
    entry = transfer._IDENTITY_PUBLICATIONS[id(original)]
    receipt = entry[3]  # _original just revalidated this receipt and its attached registry
    native = _native_definitions(source)
    by_face = {face: [] for face in faces}
    for cell in receipt.source_cells:
        if cell.face.id in by_face:
            by_face[cell.face.id].append(cell)
    result = {}
    for face, cells in by_face.items():
        if not cells:
            continue
        nodes, shell = {}, []
        for cell in cells:
            shell.append(list(cell.node_ids))
            for node, uv, xyz in zip(cell.node_ids, cell.uv, cell.xyz):
                pair = [list(uv), list(xyz)]
                if nodes.setdefault(node, pair) != pair:
                    raise MeshError('source face has multiple source corners for one node')
        ordered = sorted(nodes)
        local = {node: row for row, node in enumerate(ordered)}
        rim = []
        for use in source.faces[face].loop:
            entries = original.boundary_registry.entries(use.edge)
            rim.append([use.edge, [[float(item.key.parameter), local[int(item.node_id)]] for item in entries]])
        grid = original.grid_of_face.get(face)
        data = {'source': str(source.model_id), 'revision': source.revision,
                'face_checksum': _face_signature(source, face, native[face]), 'face': face, 'rim': rim,
                'node_rows': [nodes[n] for n in ordered],
                'cells': [[local[n] for n in cell] for cell in shell],
                'grid': None if grid is None else [[local[int(n)] for n in column] for column in grid]}
        result[str(face)] = {'data': data, 'digest': _digest(data)}
    return result


@contextmanager
def reuse_source_faces(source, candidates):
    """Propose only faces whose generation settings the caller has proved unchanged."""
    stats = {'reused_faces': set()}
    checked = {}
    incidence = {}
    if candidates:
        vertex_faces = _vertex_faces(source)
        incidence[source] = (source.revision, vertex_faces)
        native = _native_definitions(source)
        for key, record in candidates.items():
            try:
                data = record['data']
                if (record['digest'] != _digest(data) or type(data['face']) is not int
                        or str(data['face']) != key or data['source'] != str(source.model_id)
                        or data['face'] not in source.faces or not _ordinary_face(source, data['face'], vertex_faces)
                        or data['face_checksum'] != _face_signature(source, data['face'], native[data['face']])):
                    raise MeshError('retained source-face proposal differs from its source')
                # Detach from caller-owned dictionaries across generation callbacks.
                checked[data['face']] = json.loads(json.dumps(data, allow_nan=False))
            except MeshError:
                raise
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise MeshError('invalid retained source-face proposal') from error
    pending = {}
    token = _ACTIVE.set((source, checked, pending, stats, incidence, {}))
    try:
        yield stats
    finally:
        _ACTIVE.reset(token)


def retained_core(geometry, face, binding, mesh, registry, outer, protected, transform, *,
                  holes, constraints, pinned, order, declared, mapped, recombine):
    active = _ACTIVE.get()
    if active is None or face not in active[1]:
        return None
    source, proposals, _, _, incidence, lineages = active
    if geometry not in incidence or incidence[geometry][0] != geometry.revision:
        incidence[geometry] = (geometry.revision, _vertex_faces(geometry))
    if (binding is None or not binding.requires_curvilinear or binding.face_ids != {face}
            or not _ordinary_face(geometry, face, incidence[geometry][1]) or protected is None or holes
            or any(len(c) for c in constraints) or len(pinned) or order != 'linear' or mapped == 'required'):
        return None
    edges = {use.edge for use in geometry.faces[face].loop}
    if edges.intersection(declared):
        return None
    from anygeometry._local_adaptation_lineage import query_local_adaptation_lineage
    from fractions import Fraction as Q
    epoch = (source.revision, geometry.revision)
    if geometry not in lineages or lineages[geometry][0] != epoch:
        selected = tuple(sorted(proposals.keys() & geometry.faces.keys()))
        lineage = query_local_adaptation_lineage(source, geometry, selected)
        face_maps = {child: (parent, transform) for child, parent, transform in lineage.face_maps}
        edge_maps = {child: (parent, start, end) for child, parent, start, end in lineage.edge_parameter_maps}
        lineages[geometry] = (epoch, face_maps, edge_maps)
    _, face_maps, edge_maps = lineages[geometry]
    identity = ((Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)))
    if face_maps.get(face) != (face, identity):
        raise MeshError('retained source face lacks identity chart lineage')
    if any(edge_maps.get(edge) != (edge, Q(0), Q(1)) for edge in edges):
        raise MeshError('retained source face lacks identity boundary lineage')
    data = proposals[face]
    if mapped == 'mapped' and not data['grid']:
        return None
    if {edge for edge, _ in data['rim']} != edges or len(data['rim']) != len(edges):
        raise MeshError('retained source face rim is incomplete')
    old_to_new = {}
    for edge, stations in data['rim']:
        fresh = {float(entry.key.parameter): int(entry.node_id) for entry in registry.entries(edge)}
        if set(fresh) != {t for t, _ in stations} or len(fresh) != len(stations):
            raise MeshError('retained source face rim stations changed')
        for t, old in stations:
            new = fresh[t]
            if old_to_new.setdefault(old, new) != new:
                raise MeshError('retained source corner contradicts the registered station')
    if len(set(old_to_new.values())) != len(old_to_new) or set(old_to_new.values()) != set(protected.values()):
        raise MeshError('retained source rim differs from current protected inputs')
    try:
        nodes = dict(enumerate(data['node_rows']))
        if not set(old_to_new) <= nodes.keys():
            raise MeshError('retained source face has missing rows')
        row_for_new = {new:row for row,new in protected.items()}
        old_to_row = {old:row_for_new[new] for old,new in old_to_new.items()}
        for old in sorted(set(nodes).difference(old_to_new)):
            old_to_row[old] = len(old_to_row)
        uv = np.empty((len(nodes),2))
        xyz = np.empty((len(nodes),3))
        for old,row in old_to_row.items():
            uv[row], xyz[row] = nodes[old]
        if not np.isfinite(uv).all() or not np.isfinite(xyz).all():
            raise MeshError('retained source face contains nonfinite coordinates')
        if any(not np.array_equal(xyz[old_to_row[old]],mesh.nodes[new]) for old,new in old_to_new.items()):
            raise MeshError('retained source boundary changed its owner coordinates')
        cells = [[old_to_row[n] for n in c] for c in data['cells']]
        if not cells or any(len(c) not in (3,4) or len(set(c)) != len(c) for c in cells):
            raise MeshError('invalid retained source shell connectivity')
        if not recombine and any(len(c)==4 for c in cells):
            raise MeshError('retained quadrilaterals conflict with the generation request')
        triangles = np.asarray([c for c in cells if len(c)==3], dtype=int).reshape(-1,3)
        quads = np.asarray([c for c in cells if len(c)==4], dtype=int).reshape(-1,4)
        if {n for c in cells for n in c} != set(range(len(nodes))):
            raise MeshError('retained source shell has unused node rows')
        points = np.zeros((len(nodes),3))
        points[:,:2] = uv @ transform
        points[:len(outer), :2] = outer[:, :2]
        core = MeshCore(points, triangles, quads)
        if mapped == 'mapped':
            grid = np.asarray([[old_to_row[n] for n in column] for column in data['grid']], dtype=int)
            if (grid.ndim != 2 or min(grid.shape) < 2 or triangles.size or grid.size != len(nodes)
                    or set(grid.ravel()) != set(range(len(nodes)))):
                raise MeshError('retained mapped grid differs from the source rows')
            grid_cells = np.stack((grid[:-1,:-1],grid[1:,:-1],grid[1:,1:],grid[:-1,1:]),axis=-1)
            if not np.array_equal(grid_cells.reshape(-1,4),quads):
                raise MeshError('retained mapped grid differs from its current Q4 cells')
            core._parametric_tensor_grid = grid
    except MeshError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise MeshError('invalid retained source coordinates or topology') from error
    by_row = tuple(protected.get(row) for row in range(len(nodes)))
    provenance = OutputRowProvenance(by_row, tuple(sorted((n, r) for r,n in protected.items())))
    provenance.validate(core)
    core._output_row_provenance = provenance
    core._retained_source_uv = uv
    core._retained_source_xyz = xyz
    return core


def stage_reused_source_face(binding, face):
    active = _ACTIVE.get()
    if active is not None:
        active[2][id(binding.collection)] = (binding, face)


def confirm_reused_source_faces(source, mesh):
    from . import material_source_transfer as transfer
    active = _ACTIVE.get()
    if active is not None and active[0] is source:
        entry = transfer._IDENTITY_PUBLICATIONS.get(id(mesh))
        if entry is None or entry[0]() is not mesh or entry[1] is not source:
            raise MeshError('retained face confirmation requires final source publication')
        for owner, collection, chords, cells in entry[4]:
            record = active[2].get(id(collection))
            if record is not None:
                binding, face = record
                if (binding.collection is collection and binding.geometry is owner
                        and binding._curvilinear_state.get('chords') is chords
                        and binding._curvilinear_state.get('cells') is cells):
                    active[3]['reused_faces'].add(face)
