"""Exact source-edge association transport from a live owner transfer receipt."""
from fractions import Fraction
from .errors import MeshError


def source_boundaries(source, prepared, mesh, receipt, vertex_map, *, station_transport=None):
    from .boundary import GlobalEdgeBoundaryRegistry
    from .meshing_view import GeometryMeshingView
    expected_sheets = {sheet.id: {cell for use in sheet.face_use_ids
        for cell in mesh.elements_of_face.get(prepared.face_uses[use].face_id, ())}
        for sheet in prepared.sheets.values()}
    if (set(mesh.elements_of_sheet) != set(expected_sheets)
            or any(len(cells) != len(set(cells)) or set(cells) != expected_sheets[sheet]
                   or any(cell not in mesh.shells for cell in cells)
                   for sheet,cells in mesh.elements_of_sheet.items())):
        raise MeshError('prepared sheet associations differ from live face-use shell membership')
    for edge, elements in mesh.elements_of_edge.items():
        chain = mesh.offset_nodes_of_edge.get(edge) or mesh.nodes_of_edge.get(edge)
        if (chain is None or len(elements) != len(chain)-1 or len(elements) != len(set(elements))
                or any(node not in mesh.nodes for node in chain)):
            raise MeshError('prepared edge elements lack a complete station chain')
        for element,a,b in zip(elements, chain, chain[1:]):
            if (element not in mesh.beams or element in mesh.shells
                    or tuple(mesh.beams[element]) not in ((a,b),(b,a))):
                raise MeshError('prepared edge element kind or station incidence is invalid')
    maps = {edge.id: (parent, lo, hi) for edge, parent, lo, hi in receipt.edge_parameter_maps}
    traces = {edge.id: (edge, owners) for edge, owners in receipt.source_trace_ownership}
    partitions = {}
    for _edge, parent, lo, hi in receipt.edge_parameter_maps:
        if parent.model_id != source.model_id or parent.id not in source.edges or lo == hi:
            raise MeshError('source edge interval ancestry is invalid')
        partitions.setdefault(parent.id, []).append((min(lo,hi), max(lo,hi)))
    for spans in partitions.values():
        spans = sorted(spans)
        if (spans[0][0] != 0 or spans[-1][1] != 1
                or any(a[1] != b[0] for a,b in zip(spans,spans[1:]))):
            raise MeshError('source edge interval partition is incomplete or overlapping')
    stations = {(edge.id, t): (target, source_t, owners)
        for edge, t, target, source_t, owners in receipt.boundary_station_correspondence}
    aliases, affected = {}, set()
    if station_transport is not None:
        from anygeometry import validate_prepared_source_station_transport_binding
        from ._material_region_binding import curvilinear_reference_candidates
        validate_prepared_source_station_transport_binding(source, prepared, receipt, station_transport)
        declared = {}
        for model, _regions, chords, _cells in curvilinear_reference_candidates(mesh):
            if model is not prepared:
                raise MeshError('station aliases require actual generation owner')
            for interval in chords.intervals:
                for t, key, xyz in zip(interval.parameter_range, interval.station_keys, interval.xyz):
                    station = interval.edge.id, Fraction(t)
                    literal = key, tuple(xyz)
                    if station in declared and declared[station] != literal:
                        raise MeshError('generation station has conflicting literal owner keys')
                    declared[station] = literal
        for row in station_transport.rows:
            key = row.prepared_edge.id, row.prepared_parameter
            if (row.prepared_edge.model_id != prepared.model_id
                    or row.source_edge.model_id != source.model_id
                    or declared.get(key) != (row.prepared_station_key, row.xyz)):
                raise MeshError('station alias differs from literal generation station')
            if key in aliases and aliases[key] != row:
                raise MeshError('station alias repeats a conflicting prepared key')
            aliases[key] = row
            affected.add(row.source_edge.id)
        # Offset/member reference transport is not certified by this sheet-only
        # station receipt. Do not guess associations for new nominal labels.
        if any(parent.id in affected and (mesh.offset_nodes_of_edge.get(edge)
                    or mesh.elements_of_edge.get(edge)) for edge,(parent,_a,_b) in maps.items()):
            raise MeshError('station aliases do not qualify offset or member edge remapping')
    exact_identities, nominal_identities = {}, {}
    grouped, virtual, node_stations = {}, {}, set()
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(source))
    for row in mesh.boundary_registry.entries():
        edge, parameter = row.key.edge_id, Fraction(row.key.parameter)
        target = stations.get((edge, parameter))
        if (target is None or row.node_id is None or row.node_id not in mesh.nodes
                or tuple(map(float, mesh.nodes[row.node_id])) != tuple(map(float, row.point))):
            raise MeshError('closure station lacks literal owner correspondence')
        node_stations.add(row.node_id)
        parent, source_t, owners = target
        if parent is not None:
            if edge not in maps or maps[edge][0] != parent or parent.id not in source.edges:
                raise MeshError('closure source station ancestry conflicts')
            literal = tuple(map(float,row.point))
            alias = aliases.get((edge,parameter))
            if parent.id in affected:
                if (alias is None or alias.source_edge != parent or alias.source_parameter != source_t
                        or alias.xyz != literal):
                    raise MeshError('source station lacks its issued nominal alias correspondence')
                nominal, identity = alias.nominal_parameter, alias.source_station_identity
            else:
                if Fraction(float(source_t)) != source_t:
                    raise MeshError('source station is not exactly representable by the boundary registry')
                nominal, identity = float(source_t), None
            exact = parent.id, source_t
            value = identity, row.node_id, literal
            if exact in exact_identities and exact_identities[exact] != value:
                raise MeshError('duplicate exact source station has conflicting global node or literal XYZ')
            exact_identities[exact] = value
            nominal_key = parent.id, nominal
            if nominal_key in nominal_identities and nominal_identities[nominal_key] != source_t:
                raise MeshError('distinct exact source stations have a nominal alias collision')
            if 0 < source_t < 1 and nominal in (0.,1.):
                raise MeshError('interior source station aliases an endpoint')
            nominal_identities[nominal_key] = source_t
            grouped.setdefault(parent.id, []).append((source_t, nominal, row.node_id, literal))
        else:
            if edge not in traces or any(owner not in traces[edge][1] for owner in owners):
                raise MeshError('virtual trace station lacks owner source faces')
            virtual.setdefault(edge, []).append((parameter, row.node_id, tuple(map(float,row.point))))
    nodes = {}
    for edge, rows in grouped.items():
        values = {}
        for parameter, nominal, node, point in rows:
            if parameter in values and values[parameter] != (nominal, node, point):
                raise MeshError('source edge station has conflicting global identities')
            values[parameter] = nominal, node, point
        ordered = sorted(values)
        if ordered[0] != 0 or ordered[-1] != 1:
            raise MeshError('source edge station chain is incomplete')
        nodes[edge] = [values[t][1] for t in ordered]
        for t in ordered:
            nominal, node, point = values[t]
            registry.register(edge, nominal, point, node_id=node, owner=source.handle('edge',edge))
    provenance, trace_sides, trace_owners = {}, set(), {}
    for edge, rows in virtual.items():
        rows = sorted(set(rows))
        if rows[0][0] != 0 or rows[-1][0] != 1:
            raise MeshError('virtual trace station chain is incomplete')
        chain = [node for _,node,_ in rows]
        if list(mesh.nodes_of_edge.get(edge, ())) != chain:
            raise MeshError('virtual trace association differs from owner stations')
        trace_sides.update(tuple(sorted((a,b))) for a,b in zip(chain,chain[1:]))
        handle, owners = traces[edge]
        for a,b in zip(chain, chain[1:]):
            trace_owners[tuple(sorted((a,b)))] = {owner.id for owner in owners}
        provenance[f'{handle.model_id}:edge:{edge}'] = {
            'prepared_model_id': str(handle.model_id), 'prepared_edge_id': edge,
            'source_face_ids': [owner.id for owner in owners],
            'stations': [(str(t),node,point) for t,node,point in rows],
            'elements': list(mesh.elements_of_edge.get(edge, ())),
            'offset_nodes': list(mesh.offset_nodes_of_edge.get(edge, ())),
        }
    # These are mesh NODE pairs, not geometry edge identifiers.
    declared = tuple(mesh.declared_plate_junction_edges)
    known_sides = trace_sides | {tuple(sorted((a,b))) for chain in nodes.values()
                                  for a,b in zip(chain,chain[1:])}
    if any(tuple(sorted(pair)) not in known_sides for pair in declared):
        raise MeshError('declared source junction lacks owner boundary stations')
    parent_faces = {child.id: parent.id for parent, children in receipt.source_to_prepared_faces
                     for child in children}
    cell_faces = {cell: parent_faces[face] for face, cells in mesh.elements_of_face.items()
                   for cell in cells if face in parent_faces}
    incidence = {}
    for cell, corners in mesh.shells.items():
        for a,b in zip(corners, (*corners[1:],corners[0])):
            incidence.setdefault(tuple(sorted((a,b))), []).append(cell)
    declared_sides = {tuple(sorted(pair)) for pair in declared}
    for side, owners in trace_owners.items():
        cells = incidence.get(side, ())
        if {cell_faces.get(cell) for cell in cells} != owners:
            raise MeshError('virtual trace source-face incidence differs from owner receipt')
        if len(cells) > 2 and side not in declared_sides:
            raise MeshError('virtual nonmanifold trace lacks a declared mesh junction')
    tables = {'nodes_of_edge': nodes, 'declared_plate_junction_edges': declared}
    for name in ('elements_of_edge', 'offset_nodes_of_edge'):
        made = {}
        for edge, values in getattr(mesh,name).items():
            if edge in traces:
                continue
            if edge not in maps:
                raise MeshError('source edge association lacks owner ancestry')
            parent, lo, hi = maps[edge]
            made.setdefault(parent.id, []).append((min(lo,hi), list(values) if lo <= hi else list(reversed(values))))
        tables[name] = {edge:list(dict.fromkeys(v for _,row in sorted(rows) for v in row))
                        for edge,rows in made.items()}
    tables['node_of_vertex'] = {}
    virtual_vertices = {}
    for vertex, node in mesh.node_of_vertex.items():
        if vertex in vertex_map:
            tables['node_of_vertex'][vertex_map[vertex]] = node
        elif node in node_stations and vertex in prepared.vertices:
            virtual_vertices[f'{prepared.model_id}:vertex:{vertex}'] = node
        else:
            raise MeshError('generated vertex lacks owner station correspondence')
    return tables, registry, provenance, virtual_vertices, maps
