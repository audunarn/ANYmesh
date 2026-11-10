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


def _ordinary_face(model, face_id, vertex_faces, *, trim=False):
    from anygeometry.nurbs import NurbsSurface
    from ._material_region_binding import polynomial_tensor_chart, straight_source_plane
    face = model.faces[face_id]
    if ((bool(face.holes) and not (trim and len(face.holes)==1 and type(face.surface) is NurbsSurface))
            or face.parameterization is not None
            or len(face.loop) != 4 or len({use.edge for use in face.loop}) != 4):
        return False
    if (type(face.surface) is not NurbsSurface and not straight_source_plane(model, face_id)
            and polynomial_tensor_chart(model, face_id) is None):
        return False
    edges = [model.edges[u.edge] for loop in (face.loop, *face.holes) for u in loop]
    vertices = {v for e in edges for v in (e.start, e.end)}
    return all(vertex_faces[v] == {face_id} for v in vertices)


def independent_source_faces(model):
    """Whether every face is an isolated four-sided source-chart candidate.

    A positive classification selects the generation route. Source publication
    still requires a fresh, complete owner cell and rim proof.
    """
    incidence = _vertex_faces(model)
    from ._material_region_binding import straight_source_plane
    return (any(not straight_source_plane(model, face) for face in model.faces)
            and all(_ordinary_face(model, face, incidence) for face in model.faces))


def active_source_trim(source):
    """True only during generation for the currently validated source trim request."""
    active = _ACTIVE.get()
    return (active is not None and active[0] is source and bool(active[7])
            and active[4][source][0] == source.revision and source._transaction_journal is None)


def _face_signature(model, face_id, native, *, outer_only=False):
    """Pin the complete retained rectangle, its occurrences and numerical policy."""
    from anygeometry.definition_binding import definition_checksum
    face = model.faces[face_id]
    edges = tuple(model.edges[u.edge] for u in face.loop)
    vertex_ids = {v for e in edges for v in (e.start, *getattr(e.curve,'control_vertices',()), e.end)}
    vertices = tuple(model.vertices[v] for v in sorted(vertex_ids))
    uses = tuple(model.face_uses[u] for u in sorted(model._face_structural_uses.get(face_id, ())))
    sheets = tuple(model.sheets[s] for s in sorted({use.sheet_id for use in uses}))
    if outer_only:
        from dataclasses import replace
        face = replace(face, holes=())
        uses = tuple(replace(u, loops=u.loops[:1]) for u in uses)
    coedges = tuple(model.coedges[key] for u in uses for loop in u.loops for key in loop)
    return definition_checksum((model.model_id, model.tolerance, face, edges, vertices, uses, coedges, sheets, native))


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
    from ._material_region_binding import straight_source_plane
    vertex_faces = _vertex_faces(source)
    faces = [f for f in source.faces if _ordinary_face(source, f, vertex_faces)]
    if not faces or all(straight_source_plane(source, f) for f in faces):
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


def capture_source_face_identity_rows(source, mesh, face):
    """Issued permanent node and shell IDs in this face proposal's source-row order."""
    return capture_source_face_identity_tables(source, mesh, (face,))[str(face)]


def capture_source_face_identity_tables(source, mesh, faces):
    """Capture rows for all requested faces with a single live publication validation."""
    from . import material_source_transfer as transfer
    from .regional_source_numbering import _original
    original, proof = _original(mesh, source)
    grouped = {face: [] for face in faces}
    for cell in transfer._IDENTITY_PUBLICATIONS[id(original)][3].source_cells:
        if cell.face.id in grouped:
            grouped[cell.face.id].append(cell)
    original_shells = original.shells
    new_shells = mesh.shells
    result = {}
    for face, cells in grouped.items():
        ids = {tuple(original_shells[e]): e for e in original.elements_of_face[face]}
        nodes, elements = {}, []
        for c in cells:
            e = ids[c.node_ids]
            published = e if proof is None else proof.elements[e]
            elements.append(published)
            for old, new in zip(c.node_ids, new_shells[published], strict=True):
                if nodes.setdefault(old, new) != new:
                    raise MeshError('inconsistent numbered source face node rows')
        result[str(face)] = {'nodes': [nodes[n] for n in sorted(nodes)], 'cells': elements}
    return result


@contextmanager
def reuse_source_faces(source, candidates, *, point_patches=None, trim_patches=None):
    """Propose only faces whose generation settings the caller has proved unchanged."""
    stats = {'reused_faces': set(), 'patched_faces': {}, 'trimmed_faces': {}}
    trims = {int(f): json.loads(json.dumps(plan, allow_nan=False))
             for f,plan in (trim_patches or {}).items()}
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
                        or data['face'] not in source.faces or not _ordinary_face(source, data['face'], vertex_faces,
                                                                                  trim=data['face'] in trims)
                        or data['face_checksum'] != _face_signature(source, data['face'], native[data['face']],
                                                                  outer_only=data['face'] in trims)):
                    raise MeshError('retained source-face proposal differs from its source')
                # Detach from caller-owned dictionaries across generation callbacks.
                checked[data['face']] = json.loads(json.dumps(data, allow_nan=False))
            except MeshError:
                raise
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise MeshError('invalid retained source-face proposal') from error
    pending = {}
    patches = {int(f): json.loads(json.dumps(plan, allow_nan=False))
               for f, plan in (point_patches or {}).items()}
    operators = patches | trims
    if (patches.keys() & trims.keys() or not set(operators) <= checked.keys() or
            any(type(p) is not dict or type(p.get('face')) is not int or p['face'] != f for f,p in operators.items())):
        raise MeshError('a source patch requires its original face proposal')
    if trims:
        from .regional_source_trim import plan_source_face_trim
        for face, plan in trims.items():
            record = {'data': checked[face], 'digest': _digest(checked[face])}
            current = plan_source_face_trim(source, record, target_size=plan.get('target_size'),
                overrides=plan.get('overrides',()), strategy=plan.get('strategy'),
                recombine=any(len(c)==4 for c in checked[face]['cells']), quality=plan.get('quality'))
            if current is None or current != plan:
                raise MeshError('source trim differs from the current source or planned cut')
    token = _ACTIVE.set((source, checked, pending, stats, incidence, {}, patches, trims))
    try:
        yield stats
    finally:
        _ACTIVE.reset(token)


def retained_core(geometry, face, binding, mesh, registry, outer, protected, transform, *,
                  holes, constraints, pinned, order, declared, mapped, recombine,
                  size_field=None, owner_zones=0, cancellation_check=None, quality_options=None):
    active = _ACTIVE.get()
    if active is None or face not in active[1]:
        return None
    source, proposals, _, _, incidence, lineages, patches, trims = active
    if geometry not in incidence or incidence[geometry][0] != geometry.revision:
        incidence[geometry] = (geometry.revision, _vertex_faces(geometry))
    if (binding is None or not binding.requires_curvilinear or binding.face_ids != {face}
            or not _ordinary_face(geometry, face, incidence[geometry][1], trim=face in trims) or protected is None
            or (bool(holes) and not (face in trims and len(holes)==1))
            or any(len(c) for c in constraints) or len(pinned) or order != 'linear' or mapped == 'required'):
        if face in patches or face in trims:
            raise MeshError('requested source patch lacks its original generation inputs')
        return None
    definition = geometry.faces[face]
    edges = {use.edge for loop in (definition.loop, *definition.holes) for use in loop}
    if edges.intersection(declared):
        if face in patches or face in trims:
            raise MeshError('requested source patch acquired a declared junction')
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
    old_data = data
    patch = patches.get(face)
    trim = trims.get(face)
    if trim is not None:
        from .regional_source_trim import plan_source_face_trim
        from .refinement import SizeField
        if (type(size_field) is not SizeField or mapped in ('required', 'mapped')
                or size_field.zones[:len(size_field.zones)-owner_zones]):
            raise MeshError('source trim conflicts with the active meshing request')
        quality = trim.get('quality')
        if trim.get('annulus'):
            from .surface_mesh import SurfaceMeshOptions
            actual = SurfaceMeshOptions()
            if quality_options is not None:
                policy = quality_options.quality_policy
                actual = SurfaceMeshOptions(min_scaled_jacobian=policy.minimum_scaled_jacobian,
                    max_aspect_ratio=policy.maximum_aspect_ratio,min_angle=policy.minimum_angle,
                    max_angle=policy.maximum_angle,max_warpage=policy.maximum_warpage,
                    max_element_growth=quality_options.max_element_growth)
            if any(getattr(actual,key) != value for key,value in quality.items()):
                raise MeshError('source trim quality differs from the actual generation request')
            quality = {key:getattr(actual,key) for key in ('min_scaled_jacobian','max_aspect_ratio',
                       'min_angle','max_angle','max_warpage','max_element_growth')}
        current = plan_source_face_trim(source, {'data':data, 'digest':_digest(data)},
            target_size=size_field.target_size, overrides=trim.get('overrides', ()),
            strategy=trim.get('strategy'), recombine=recombine, quality=quality,
            cancellation_check=cancellation_check)
        if current is not None and trim.get('annulus'):
            current['quality'] = trim['quality']
        if current is None or current != trim:
            raise MeshError('source trim differs from the current source or planned cut')
        removed = set(trim['replaced_cell_rows'])
        kept = {k: c for k,c in enumerate(data['cells']) if k not in removed}
        data = dict(data, cells=[*kept.values(), *trim.get('new_cells',())],
                    node_rows=[*data['node_rows'], *trim.get('new_rows',())],
                    rim=data['rim'] + trim['hole_rim'], grid=None)
    if patch is not None:
        from .regional_source_patch import _subdivide, _support, plan_source_face_point_patch
        from .refinement import SizeField
        if type(size_field) is not SizeField or mapped in ('required', 'mapped') or not recombine:
            raise MeshError('source patch conflicts with current meshing strategy or size field')
        zones = size_field.zones[:len(size_field.zones) - owner_zones]
        if len(zones) != 1:
            raise MeshError('source patch requires exactly one current spatial refinement')
        current = plan_source_face_point_patch(source, {'data':data, 'digest':_digest(data)}, zones[0],
            target_size=size_field.target_size, strategy=patch.get('strategy'), recombine=recombine,
            cancellation_check=cancellation_check)
        if current is None or current != patch:
            raise MeshError('source patch differs from its current source, field or operator')
        data, kept, _ = _subdivide(data, current['replaced_cell_rows'], _support(source, face))
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
        if trim is not None:
            nodes = {n: nodes[n] for cell in data['cells'] for n in cell}
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
        boundary = np.vstack((outer, *holes))
        points[:len(boundary), :2] = boundary[:, :2]
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
    if patch is not None:
        core._source_patch_rows = (patch, {n: old_to_row[n] for n in range(len(old_data['node_rows']))}, kept)
    if trim is not None:
        core._source_patch_rows = (trim, old_to_row, kept)
    return core


def stage_reused_source_face(binding, face, core, core_to_global, mesh):
    active = _ACTIVE.get()
    if active is not None:
        rows = getattr(core, '_source_patch_rows', None)
        evidence = None
        staged_cells = {}
        if rows is not None:
            patch, old_to_row, kept = rows
            published = {tuple(mesh.tris[e] if e in mesh.tris else mesh.quads[e]): e for e in mesh.elements_of_face[face]}
            nodes = {n: core_to_global[row] for n,row in old_to_row.items()}
            cells = {k: published[tuple(nodes[n] for n in c)] for k,c in kept.items()}
            evidence = {'old_node_rows': nodes, 'kept_cell_rows': cells,
                        'interface_rows': list(patch['interface_rows']),
                        'replaced_cell_rows': list(patch['replaced_cell_rows'])}
            if patch.get('annulus'):
                fixed = ({n for c in kept.values() for n in c} | set(patch['interface_rows'])
                         | set(patch['hole_cell_interface_rows']))
                for c in patch['new_cells']:
                    conn = tuple(nodes[n] for n in c)
                    staged_cells[published[conn]] = conn
                evidence.update(annulus=True, old_node_rows={n:nodes[n] for n in fixed},
                    new_node_rows={n:g for n,g in nodes.items() if n >= patch['parent_node_count']},
                    new_cell_ids=list(staged_cells), hole_cell_interface_rows=list(patch['hole_cell_interface_rows']))
        active[2][id(binding.collection)] = (binding, face, evidence, staged_cells)


def confirm_reused_source_faces(source, mesh):
    from . import material_source_transfer as transfer
    active = _ACTIVE.get()
    if active is not None and active[0] is source:
        entry = transfer._IDENTITY_PUBLICATIONS.get(id(mesh))
        if entry is None or entry[0]() is not mesh or entry[1] is not source:
            raise MeshError('retained face confirmation requires final source publication')
        reused, patched, trimmed = set(), {}, {}
        for owner, collection, chords, cells in entry[4]:
            record = active[2].get(id(collection))
            if record is not None:
                binding, face, evidence, staged_cells = record
                if (binding.collection is collection and binding.geometry is owner
                        and binding._curvilinear_state.get('chords') is chords
                        and binding._curvilinear_state.get('cells') is cells):
                    if evidence is None:
                        reused.add(face)
                    else:
                        proposal = active[1][face]
                        nodes = evidence['old_node_rows']
                        shell = mesh.shells
                        face_elements = set(mesh.elements_of_face[face])
                        if face in active[7]:
                            hole_rim = active[7][face]['hole_rim']
                            if (face_elements != set(evidence['kept_cell_rows'].values()) | staged_cells.keys()
                                    or any(shell[e] != c for e,c in staged_cells.items()) or any(
                                    [(float(s.key.parameter), s.node_id) for s in mesh.boundary_registry.entries(e)] !=
                                    [(t, (evidence['new_node_rows'] if staged_cells else nodes)[n])
                                     for t,n in stations] for e,stations in hole_rim)):
                                raise MeshError('source trim changed the qualified material or hole station table')
                        if (any(e not in face_elements or shell[e] != tuple(nodes[n] for n in proposal['cells'][k])
                                for k,e in evidence['kept_cell_rows'].items()) or
                            any(not np.array_equal(mesh.nodes[node], proposal['node_rows'][k][1])
                                for k,node in nodes.items())):
                            raise MeshError('fixed source patch exterior changed at final publication')
                        (trimmed if face in active[7] else patched)[face] = evidence
        if set(patched) != set(active[6]) or set(trimmed) != set(active[7]):
            raise MeshError('final source publication lacks the requested fixed-interface patch')
        active[3]['reused_faces'].update(reused)
        active[3]['patched_faces'].update(patched)
        active[3]['trimmed_faces'].update(trimmed)
