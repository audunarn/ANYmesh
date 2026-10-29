"""Owner-bound source transfer and narrow unchanged-clone mesh publication.

The public query grants no publication/S3 admission. The private publisher
requires the original live mesh, generation receipts and exact identity maps.
"""
from .errors import MeshError
from ._material_region_binding import face_requires_curvilinear, region_requires_curvilinear  # patched by routing tests

_IDENTITY_PUBLICATIONS = {}



class _ClosurePublicationQualification:
    """Invocation-local opaque identity; copying/serde issues no authority."""
    pass


from weakref import WeakKeyDictionary
_CLOSURE_PUBLICATION_QUALIFICATIONS = WeakKeyDictionary()


def _closure_publication_operands(source, closure):
    from anygeometry import GeometryModel, ModelClosure
    if (not isinstance(source, GeometryModel) or type(closure) is not ModelClosure
            or not isinstance(closure.working_model, GeometryModel)):
        raise MeshError('closure publication requires a live model and ModelClosure')


def _closure_selection_pin(closure):
    """Detached scalar values: frozen handles/UUIDs can still be raw-edited."""
    from uuid import UUID
    from anygeometry import EntityHandle
    def uuid(value):
        if type(value) is not UUID or type(value.int) is not int:
            raise MeshError('closure publication UUID definition is malformed')
        return type(value), value.int
    def handle(value):
        if (type(value) is not EntityHandle or type(value.kind) is not str
                or type(value.id) is not int or value.id <= 0):
            raise MeshError('closure publication handle definition is malformed')
        return (type(value), uuid(value.model_id), type(value.kind), value.kind,
                type(value.id), value.id)
    try:
        if type(closure.source_revision) is not int or closure.source_revision < 0:
            raise MeshError('closure publication source revision is malformed')
        return (tuple((handle(key),handle(value)) for key,value in closure.source_to_work.items()),
                tuple((handle(key),handle(value)) for key,value in closure.work_to_source.items()),
                tuple(handle(value) for value in closure.source_handles),
                uuid(closure.source_model_id), type(closure.source_revision), closure.source_revision)
    except (AttributeError,TypeError,ValueError) as error:
        raise MeshError('closure publication selection is malformed') from error


def qualify_model_closure_source_publication(source, closure, *, cancellation_check=None):
    """Qualify an uncut extraction before owner imprint and detached mesh attempts.

    The opaque token retains no serialized authority. Final publication still
    requires current owner cell transfers from both the source and this work.
    """
    from anygeometry import qualify_model_closure_source, to_dict
    _closure_publication_operands(source, closure)
    before = to_dict(source)
    pins = _closure_selection_pin(closure)
    objects = (closure.source_to_work, closure.work_to_source, closure.source_handles)
    work = closure.working_model
    qualify_model_closure_source(source, closure, cancellation_check=cancellation_check)
    if (to_dict(source) != before or closure.working_model is not work
            or any(a is not b for a, b in zip(objects,
                (closure.source_to_work, closure.work_to_source, closure.source_handles)))
            or _closure_selection_pin(closure) != pins):
        raise MeshError('closure source qualification operands changed')
    token = _ClosurePublicationQualification()
    _CLOSURE_PUBLICATION_QUALIFICATIONS[token] = (source, work, objects, pins, before)
    return token


def _closure_publication_qualification(source, closure, token, *, check_source=False):
    try:
        entry = (_CLOSURE_PUBLICATION_QUALIFICATIONS.get(token)
                 if type(token) is _ClosurePublicationQualification else None)
    except TypeError:
        entry = None
    if entry is None or entry[0] is not source:
        raise MeshError('closure publication qualification is forged or belongs to another source')
    _source, work, objects, pins, document = entry
    if (any(a is not b for a, b in zip(objects,
            (closure.source_to_work, closure.work_to_source, closure.source_handles)))
            or _closure_selection_pin(closure) != pins):
        raise MeshError('closure publication qualification maps or selection changed')
    if check_source:
        from anygeometry import to_dict
        if to_dict(source) != document:
            raise MeshError('closure publication qualification source changed')
    return work

def requires_material_source_transfer(geometry, mesh):
    from anygeometry import ExtrudedSurface, CoonsSurface, query_material_surface_regions, GeometryError
    from anygeometry.surfaces import Plane, Cylinder, Cone, RuledSurface
    from anygeometry.curves import Straight, Arc
    from anygeometry.exact_curves import EXACT_CURVES
    
    from ._material_region_binding import curvilinear_reference_candidates
    if bool(curvilinear_reference_candidates(mesh)):
        return True
    face_ids = tuple(face for face, cells in getattr(mesh, 'elements_of_face', {}).items() if cells)
    if not face_ids:
        face_ids = tuple(geometry.faces)
    if any(face not in geometry.faces for face in face_ids):
        return True
    if any(face_requires_curvilinear(geometry, face) for face in face_ids):
        return True
    for key in face_ids:
        face = geometry.faces[key]
        if face.surface is not None and not isinstance(
                face.surface, (Plane, Cylinder, Cone, RuledSurface, ExtrudedSurface, CoonsSurface)):
            return True
        if any(not isinstance(geometry.edges[use.edge].curve, (Straight, Arc, *EXACT_CURVES))
               for loop in (face.loop, *face.holes) for use in loop):
            return True
    candidates=tuple(face.id for face in (geometry.faces[key] for key in face_ids)
                     if isinstance(face.surface,(ExtrudedSurface,CoonsSurface)))
    if candidates:
        try:
            regions=query_material_surface_regions(geometry,candidates)
        except GeometryError:
            return True
        return any(region_requires_curvilinear(region) for region in regions.regions)
    return False


def _publication_pin(mesh):
    from dataclasses import fields, is_dataclass
    from uuid import UUID
    import numpy as np
    def literal(value):
        kind = type(value)
        if value is None or kind in (bool, int, str):
            return kind, value
        if kind is float:
            return kind, value.hex()
        if kind is UUID:
            return kind, value.hex
        if kind is np.ndarray:
            if value.dtype.hasobject:
                raise MeshError('source publication has nonliteral mesh arrays')
            return kind, value.dtype.str, value.shape, value.tobytes()
        if isinstance(value, np.generic):
            return kind, value.dtype.str, value.tobytes()
        if kind in (tuple, list):
            return kind, tuple(literal(item) for item in value)
        if kind is dict:
            return kind, tuple(sorted(((literal(k), literal(v)) for k, v in value.items()), key=repr))
        if is_dataclass(value):
            return kind, tuple((field.name, literal(getattr(value, field.name))) for field in fields(value))
        raise MeshError('source publication has nonliteral mesh associations')
    # Preserve literal types: persistence intentionally coerces integer IDs.
    # Diagnostics, generation requests and later S3 records are not source
    # correspondence authority; final publication separately guards metadata.
    names = ('geometry_model_id', 'geometry_revision', 'order', 'nodes', 'quads',
        'tris', 'beams', 'couplings', 'node_of_vertex', 'nodes_of_edge',
        'offset_nodes_of_edge', 'grid_of_face', 'block_grids_of_face',
        'elements_of_face', 'elements_of_edge', 'elements_of_sheet',
        'elements_of_member', 'nodes_of_member', 'activity', 'thickness_of_face',
        'declared_plate_junction_edges')
    seeding = getattr(mesh, 'seeding', None)
    field = None if seeding is None else seeding.size_field
    if field is not None and not hasattr(field, '__dict__'):
        raise MeshError('source publication has an unsupported seeding size field')
    seed_pin = None if seeding is None else (literal(seeding.divisions), literal(seeding.classes),
        None if field is None else (type(field), literal(field.__dict__)))
    provenance = getattr(mesh, 'structural_preparation', {}).get('prepared_boundary_provenance')
    return tuple((name, literal(getattr(mesh, name))) for name in names) + (
        ('seeding', seed_pin), ('prepared_boundary_provenance', literal(provenance)))


def _publish_identity_source_transfer(source, prepared, mesh, registry):
    """Publish a genuine unchanged clone using owner-declared literal stations.

    This first consumer route deliberately requires identity face/edge maps;
    split descendants and virtual new traces require the general remapper.
    """
    from weakref import ref
    from .boundary import GlobalEdgeBoundaryRegistry
    from .meshing_view import GeometryMeshingView
    from ._material_region_binding import (curvilinear_reference_candidates,
        curvilinear_reference_registry, boundary_registry_pin)
    if getattr(mesh, 'boundary_registry', None) is not registry:
        raise MeshError('source publication requires the actual prepared registry')
    receipt = validate_material_source_transfer_for_mesh(source, prepared, mesh)
    if any(tuple(child.id for child in children) != (parent.id,)
           for parent, children in receipt.source_to_prepared_faces):
        raise MeshError('source publication requires qualified general descendant remapping')
    edge_maps = {current.id: (original.id, lo, hi)
                 for current, original, lo, hi in receipt.edge_parameter_maps}
    if receipt.source_trace_ownership or any(
            edge != original or lo != 0 or hi != 1
            for edge, (original, lo, hi) in edge_maps.items()):
        raise MeshError('source publication requires qualified general edge station transport')
    stations = {(edge.id, parameter): (target, source_t)
                for edge, parameter, target, source_t, _faces
                in receipt.boundary_station_correspondence}
    made = GlobalEdgeBoundaryRegistry(GeometryMeshingView(source))
    for entry in registry.entries():
        edge, parameter = entry.key.edge_id, entry.key.parameter
        target = stations.get((edge, parameter))
        if (edge_maps.get(edge) != (edge, 0, 1) or target is None
                or target[0] != source.handle('edge', edge) or target[1] != parameter
                or entry.node_id is None
                or tuple(map(float, mesh.nodes[entry.node_id])) != tuple(map(float, entry.point))):
            raise MeshError('source publication station lacks exact owner correspondence')
        made.register(edge, parameter, entry.point, node_id=entry.node_id,
                      owner=source.handle('edge', edge))
    # No closest-point projection or geometric identity discovery occurs here.
    candidates = curvilinear_reference_candidates(mesh)
    old_registry_authority = curvilinear_reference_registry(mesh)
    old_identity = (mesh.geometry_model_id, mesh.geometry_revision)
    mesh.geometry_model_id, mesh.geometry_revision = source.model_id, source.revision
    mesh.boundary_registry = made
    key = id(mesh)
    def cleanup(reference):
        current = _IDENTITY_PUBLICATIONS.get(key)
        if current is not None and current[0] is reference:
            _IDENTITY_PUBLICATIONS.pop(key, None)
    try:
        entry = (ref(mesh, cleanup), source, prepared, receipt, candidates,
                 old_registry_authority, registry, made, boundary_registry_pin(made),
                 _publication_pin(mesh))
        _IDENTITY_PUBLICATIONS[key] = entry
        _validate_identity_source_publication(source, prepared, mesh, made)
    except BaseException:
        _IDENTITY_PUBLICATIONS.pop(key, None)
        mesh.geometry_model_id, mesh.geometry_revision = old_identity
        mesh.boundary_registry = registry
        raise
    return made


def _validate_identity_source_publication(source, prepared, mesh, registry):
    from anygeometry import validate_prepared_material_source_transfer_binding
    from ._material_region_binding import (curvilinear_reference_candidates,
        curvilinear_reference_registry, boundary_registry_pin)
    entry = _IDENTITY_PUBLICATIONS.get(id(mesh))
    if (entry is None or entry[0]() is not mesh or entry[1] is not source
            or entry[2] is not prepared or entry[7] is not registry):
        raise MeshError('source publication lacks generation-owned transfer authority')
    _, _, _, receipt, candidates, original, old_registry, _, registry_pin, mesh_pin = entry[:10]
    if (curvilinear_reference_candidates(mesh) is not candidates
            or curvilinear_reference_registry(mesh) is not original
            or original[0]() is not old_registry
            or boundary_registry_pin(old_registry) != original[1]
            or getattr(mesh, 'boundary_registry', None) is not registry
            or boundary_registry_pin(registry) != registry_pin
            or _publication_pin(mesh) != mesh_pin):
        raise MeshError('source publication cells, stations or associations changed')
    validate_prepared_material_source_transfer_binding(source, prepared, receipt,
        cell_receipts=tuple(item[3] for item in candidates))
    if len(entry) > 10 and entry[10] is not None:
        from anygeometry import validate_prepared_source_station_transport_binding
        validate_prepared_source_station_transport_binding(source, prepared, receipt, entry[10])
    return receipt


def _validated_curvilinear_reference_uv(geometry, mesh):
    """Actual owner-proved cell reference points for physical normal evaluation."""
    from anygeometry import validate_prepared_material_curvilinear_cells_binding
    from ._material_region_binding import (curvilinear_reference_candidates,
        curvilinear_reference_registry, boundary_registry_pin,
        curvilinear_boundary_associations, face_requires_curvilinear)
    faces = {face for face, elements in mesh.elements_of_face.items()
             if elements and face_requires_curvilinear(geometry, face)}
    publication = _IDENTITY_PUBLICATIONS.get(id(mesh))
    if publication is not None and publication[0]() is mesh:
        transferred = _validate_identity_source_publication(
            geometry, publication[2], mesh, mesh.boundary_registry)
        faces = {cell.face.id for cell in transferred.source_cells}
    if not faces:
        # Copies and persistence deliberately lose live proof identity. In a
        # mixed curved closure, ordinary labels alone cannot prove that the
        # cells came only from its planar subset.
        if mesh.shells and any(face_requires_curvilinear(geometry, face)
                               for face in geometry.faces):
            raise MeshError('mixed curvilinear closure requires actual owner-proved cell authority')
        if (curvilinear_reference_candidates(mesh)
                or (id(mesh) in _IDENTITY_PUBLICATIONS
                    and _IDENTITY_PUBLICATIONS[id(mesh)][0]() is mesh)):
            raise MeshError('curvilinear owner-proved cells lost their face associations')
        return {}
    if mesh.geometry_model_id != geometry.model_id or mesh.geometry_revision != geometry.revision:
        raise MeshError('curvilinear physical normals require the current geometry namespace')
    pin = _publication_pin(mesh)
    publication = _IDENTITY_PUBLICATIONS.get(id(mesh))
    if publication is not None and publication[0]() is mesh:
        receipt = _validate_identity_source_publication(
            geometry, publication[2], mesh, mesh.boundary_registry)
        cells, references = receipt.source_cells, receipt.reference_uv_by_cell
    else:
        candidates = curvilinear_reference_candidates(mesh)
        registry = curvilinear_reference_registry(mesh)
        attached = getattr(mesh, 'boundary_registry', None)
        if (not candidates or any(item[0] is not geometry for item in candidates)
                or registry is None or registry[0]() is None
                or registry[0]().view.source is not geometry
                or (attached is not None and attached is not registry[0]())
                or boundary_registry_pin(registry[0]()) != registry[1]
                or curvilinear_boundary_associations(mesh) != registry[2]):
            raise MeshError('curvilinear physical normals require actual owner cell authority')
        cells, references = [], []
        for _owner, regions, _chords, receipt in candidates:
            validate_prepared_material_curvilinear_cells_binding(geometry, regions, receipt)
            cells.extend(receipt.cells)
            references.extend(receipt.reference_uv_by_cell)
    expected, uv = {}, {}
    for cell in cells:
        key = (cell.face.id, cell.node_ids)
        if key in expected:
            raise MeshError('curvilinear physical normals repeat owner cells')
        expected[key] = cell.xyz
    for face, nodes, point in references:
        key = (face.id, nodes)
        if key in uv:
            raise MeshError('curvilinear physical normals repeat reference points')
        uv[key] = point
    actual, result, seen = {}, {}, set()
    for face in faces:
        for element in mesh.elements_of_face[face]:
            if element in seen or (element in mesh.tris) == (element in mesh.quads):
                raise MeshError('curvilinear physical normal cell identity is ambiguous')
            seen.add(element)
            nodes = tuple(mesh.tris[element] if element in mesh.tris else mesh.quads[element])
            key = (face, nodes)
            if key in actual or key not in uv:
                raise MeshError('curvilinear physical normal lacks one owner reference point')
            actual[key] = tuple(tuple(map(float, mesh.nodes[node])) for node in nodes)
            result[element] = uv[key]
    if actual != expected or _publication_pin(mesh) != pin:
        raise MeshError('curvilinear physical normals differ from owner-proved cells')
    return result


def _qualify_identity_source_s3(source, prepared, mesh):
    """Recompute source FaceUse normals and the unchanged S3 floor afresh."""
    from .s3_production import prepare_qualified_s3_mesh, S3OwnerAuthorityError
    _validate_identity_source_publication(source, prepared, mesh, mesh.boundary_registry)
    before = _publication_pin(mesh)
    made, record = prepare_qualified_s3_mesh(mesh, source, target_policy=None)
    if _publication_pin(mesh) != before or _publication_pin(made) != before:
        raise S3OwnerAuthorityError('source S3 needs new owner proof after physical mesh repair')
    _validate_identity_source_publication(source, prepared, mesh, mesh.boundary_registry)
    record['authority_model'].update(source_model_id=str(source.model_id),
        source_revision=source.revision)
    if record.get('status') == 'ADMITTED':
        record['authority_model']['scope'] = 'SOURCE_GEOMETRY_ORIENTED_SHEET_FACE_USE'
    return record


def validate_material_source_transfer_for_mesh(source, prepared, mesh, *,
        cancellation_check=None):
    """Return an owner transfer receipt for the literal curvilinear mesh cells.

Both face associations and XYZ are checked against the original owner-proved
cells. Copies, persistence and caller-authored face maps supply no authority.
The input mesh must still use the prepared model's association namespace.
"""
    from anygeometry import (
        query_prepared_material_source_transfer,
        validate_prepared_material_source_transfer_binding,
    )
    from .mesh import Mesh
    from ._authored_component_stage import _mesh_digest
    from ._material_region_binding import (
        boundary_registry_pin, curvilinear_reference_candidates,
        face_requires_curvilinear, curvilinear_reference_registry,
        curvilinear_boundary_associations,
    )
    if type(mesh) is not Mesh:
        raise MeshError('source transfer requires the actual canonical mesh')
    candidates = curvilinear_reference_candidates(mesh)
    if not candidates or any(item[0] is not prepared for item in candidates):
        raise MeshError('source transfer lacks generation-owned prepared cell receipts')
    if (mesh.geometry_model_id != prepared.model_id
            or mesh.geometry_revision != prepared.revision):
        raise MeshError('source transfer requires prepared-current mesh associations')
    registry = getattr(mesh, 'boundary_registry', None)
    if registry is None or registry.view.source is not prepared:
        raise MeshError('source transfer requires the actual prepared boundary registry')
    registry_authority = curvilinear_reference_registry(mesh)
    if (registry_authority is None or registry_authority[0]() is not registry
            or boundary_registry_pin(registry) != registry_authority[1]
            or curvilinear_boundary_associations(mesh) != registry_authority[2]):
        raise MeshError('source transfer boundary stations differ from generation authority')
    registry.view.assert_current(prepared)
    mesh_pin, registry_pin = _mesh_digest(mesh), boundary_registry_pin(registry)

    def current():
        if (curvilinear_reference_candidates(mesh) is not candidates
                or curvilinear_reference_registry(mesh) is not registry_authority
                or _mesh_digest(mesh) != mesh_pin
                or getattr(mesh, 'boundary_registry', None) is not registry
                or boundary_registry_pin(registry) != registry_pin):
            raise MeshError('source transfer mesh or registry changed during qualification')

    def check(phase):
        current()
        cancelled = cancellation_check(phase) if cancellation_check is not None else False
        current()
        return cancelled

    receipts = tuple(item[3] for item in candidates)
    expected = {}
    for receipt in receipts:
        for cell in receipt.cells:
            key = (cell.face.id, cell.node_ids)
            if key in expected:
                raise MeshError('source transfer has duplicate owner cell membership')
            expected[key] = cell.xyz
    relevant_faces = {face for face, elements in mesh.elements_of_face.items()
                      if elements and face_requires_curvilinear(prepared, face)}
    actual, element_owners = {}, {}
    for face in relevant_faces:
        for element in mesh.elements_of_face[face]:
            if element in element_owners:
                raise MeshError('source transfer has duplicate mesh cell membership')
            element_owners[element] = face
            if (element in mesh.tris) == (element in mesh.quads):
                raise MeshError('source transfer mesh cell kind is ambiguous')
            nodes = tuple(mesh.tris[element] if element in mesh.tris else mesh.quads[element])
            if len(nodes) != (3 if element in mesh.tris else 4):
                raise MeshError('source transfer requires linear shell cells')
            key = (face, nodes)
            if key in actual:
                raise MeshError('source transfer repeats a literal mesh cell')
            actual[key] = tuple(tuple(map(float, mesh.nodes[node])) for node in nodes)
    if actual != expected:
        raise MeshError('source transfer differs from owner-proved mesh coverage or XYZ')
    current()
    receipt = query_prepared_material_source_transfer(source, prepared, receipts,
        expected_source_revision=source.revision,
        expected_prepared_revision=prepared.revision, cancellation_check=check)
    validate_prepared_material_source_transfer_binding(source, prepared, receipt,
        cell_receipts=receipts, cancellation_check=check)
    current()
    return receipt


def publish_model_closure_mesh_to_source(source, closure, mesh, *, cancellation_check=None,
                                         qualification=None):
    """Publish a live owner-proved closure mesh in its original source namespace.

    Reconstructing the closure authenticates its structural correspondence;
    geometry's live source-cell transfer independently proves the actual cells
    and boundary stations. Neither transported maps nor a prepared S3 record
    supplies either proof. Publication and fresh source S3 are rollback-safe.
    """
    from copy import copy, deepcopy
    from weakref import ref
    from numbers import Integral
    from anygeometry import qualify_model_closure_source
    from .mesh import Mesh
    from .boundary import GlobalEdgeBoundaryRegistry
    from .meshing_view import GeometryMeshingView
    from ._material_region_binding import (curvilinear_reference_candidates,
        curvilinear_reference_registry, boundary_registry_pin)

    _closure_publication_operands(source, closure)
    if type(mesh) is not Mesh:
        raise MeshError('closure publication requires the actual canonical mesh')
    work = closure.working_model
    if qualification is None and (mesh.geometry_model_id != work.model_id
                                   or mesh.geometry_revision != work.revision):
        raise MeshError('closure publication requires working-current mesh associations')
    old_publication = _IDENTITY_PUBLICATIONS.get(id(mesh))
    if old_publication is not None:
        if old_publication[0]() is not mesh or (qualification is None and old_publication[1] is not work):
            raise MeshError('closure publication requires the original live generation publication')
        prepared = old_publication[2]
        _validate_identity_source_publication(old_publication[1], prepared, mesh, mesh.boundary_registry)
    else:
        candidates = curvilinear_reference_candidates(mesh)
        if not candidates or any(item[0] is not candidates[0][0] for item in candidates):
            raise MeshError('closure publication requires the original live generation cells')
        prepared = candidates[0][0]
        if (mesh.geometry_model_id != prepared.model_id
                or mesh.geometry_revision != prepared.revision):
            raise MeshError('closure publication requires generation-current mesh associations')
        if qualification is None and prepared is not work:
            raise MeshError('closure publication requires qualified detached generation lineage')
    from ._publication_snapshot import PublicationSnapshot
    from ._material_region_binding import _CURVILINEAR_MESH_REFERENCES
    entry_pin = _publication_pin(mesh)
    old_references = _CURVILINEAR_MESH_REFERENCES.get(id(mesh))
    snapshot = PublicationSnapshot(mesh, old_publication, old_references)

    def guarded_callback(phase):
        if closure.working_model is not work:
            raise MeshError('closure working model changed during publication')
        if qualification is not None:
            _closure_publication_qualification(source, closure, qualification)
        if cancellation_check is None:
            return False
        before_publication = _IDENTITY_PUBLICATIONS.get(id(mesh))
        before_references = _CURVILINEAR_MESH_REFERENCES.get(id(mesh))
        before = PublicationSnapshot(mesh, before_publication, before_references)
        result = cancellation_check(phase)
        if closure.working_model is not work:
            raise MeshError('closure working model changed during publication')
        if qualification is not None:
            _closure_publication_qualification(source, closure, qualification)
        if (before.changed() or _IDENTITY_PUBLICATIONS.get(id(mesh)) is not before_publication
                or _CURVILINEAR_MESH_REFERENCES.get(id(mesh)) is not before_references):
            raise MeshError('closure source callback changed owned mesh or registry state')
        return result

    def check(phase):
        if guarded_callback(phase):
            raise MeshError('closure source publication cancelled')

    # A previous unchanged-clone publication changed only the model namespace
    # and registry. Restore that authenticated generation namespace privately
    # while qualifying the transitive source route. All later changes replace
    # tables, leaving the caller's original table objects available to rollback.
    try:
        if qualification is None:
            qualify_model_closure_source(source, closure, cancellation_check=guarded_callback)
            qualified_work = work
        else:
            qualified_work = _closure_publication_qualification(
                source, closure, qualification, check_source=True)
        check('closure source publication start')
        if _publication_pin(mesh) != entry_pin:
            raise MeshError('closure source mesh changed before publication')
        mesh.geometry_model_id, mesh.geometry_revision = prepared.model_id, prepared.revision
        if old_publication is not None:
            mesh.boundary_registry = old_publication[6]
        receipt = validate_material_source_transfer_for_mesh(
            source, prepared, mesh, cancellation_check=guarded_callback)
        # A same-UUID recipe clone is not ownership. Require independent live
        # transfers from BOTH the originally qualified work and current closure.
        working_transfers = []
        if qualification is not None:
            for working_owner in dict.fromkeys((qualified_work, work)):
                if prepared is not working_owner:
                    working_transfers.append((working_owner, validate_material_source_transfer_for_mesh(
                        working_owner, prepared, mesh, cancellation_check=guarded_callback)))
        face_map = {child.id: parent.id for parent, children in receipt.source_to_prepared_faces
                    for child in children}
        maps = {kind: {} for kind in ('vertex', 'sheet', 'member')}
        for target, original in closure.work_to_source.items():
            if target.kind in maps:
                maps[target.kind][target.id] = original.id

        def remap(table, mapping, *, aggregate=False):
            made = {}
            for key, value in table.items():
                if isinstance(key, bool) or not isinstance(key, Integral) or int(key) not in mapping:
                    raise MeshError('closure mesh association lacks an owner source correspondence')
                target = mapping[int(key)]
                if aggregate:
                    made.setdefault(target, []).extend(value)
                elif target in made:
                    if made[target] != value:
                        raise MeshError('closure source association is ambiguous')
                else:
                    made[target] = value
            return made

        tables = {}
        for name in ('elements_of_face', 'thickness_of_face'):
            tables[name] = remap(getattr(mesh, name), face_map, aggregate=name == 'elements_of_face')
        if mesh.grid_of_face or mesh.block_grids_of_face:
            raise MeshError('split source publication does not qualify structured grid transport')
        tables['grid_of_face'], tables['block_grids_of_face'] = {}, {}
        station_transport = None
        from fractions import Fraction
        if any(parent is not None and Fraction(float(t)) != t
               for _edge,_p,parent,t,_owners in receipt.boundary_station_correspondence):
            from anygeometry import (query_prepared_source_station_transport,
                validate_prepared_source_station_transport_binding, is_cancellation_signal)
            alias_callback = (cancellation_check if cancellation_check is None
                or is_cancellation_signal(cancellation_check) else guarded_callback)
            alias_pin, alias_registry = _publication_pin(mesh), mesh.boundary_registry
            alias_registry_pin = boundary_registry_pin(alias_registry)
            alias_publication = _IDENTITY_PUBLICATIONS.get(id(mesh))
            alias_references = _CURVILINEAR_MESH_REFERENCES.get(id(mesh))
            station_transport = query_prepared_source_station_transport(
                source, prepared, receipt, cancellation_check=alias_callback)
            validate_prepared_source_station_transport_binding(source, prepared, receipt,
                station_transport, cancellation_check=alias_callback)
            if (_publication_pin(mesh) != alias_pin or mesh.boundary_registry is not alias_registry
                    or boundary_registry_pin(alias_registry) != alias_registry_pin
                    or _IDENTITY_PUBLICATIONS.get(id(mesh)) is not alias_publication
                    or _CURVILINEAR_MESH_REFERENCES.get(id(mesh)) is not alias_references):
                raise MeshError('station alias qualification changed literal mesh or registry')
        from ._source_boundary_publication import source_boundaries
        boundary_tables, registry, virtual, virtual_vertices, edge_maps = source_boundaries(
            source, prepared, mesh, receipt, maps['vertex'], station_transport=station_transport)
        tables.update(boundary_tables)
        tables['elements_of_sheet'] = remap(mesh.elements_of_sheet, maps['sheet'])
        for name in ('elements_of_member', 'nodes_of_member'):
            tables[name] = remap(getattr(mesh, name), maps['member'])
        if mesh.seeding is not None:
            from ._source_seeding_publication import project_source_seeding
            tables['seeding'] = project_source_seeding(
                source, mesh, receipt, edge_maps, boundary_tables['nodes_of_edge'])
        # Stage metadata before changing any caller-owned association table.
        preparation = deepcopy(mesh.structural_preparation)
        preparation['source_to_working_faces'] = {
            str(parent.id): [child.id for child in children]
            for parent, children in receipt.source_to_prepared_faces}
        preparation['source_to_working_edges'] = {}
        for current, original, _lo, _hi in receipt.edge_parameter_maps:
            preparation['source_to_working_edges'].setdefault(str(original.id), []).append(current.id)
        preparation['prepared_boundary_provenance'] = {
            'traces': virtual, 'vertices': virtual_vertices,
            'source_model_id': str(source.model_id),
            'prepared_model_id': str(prepared.model_id),
            'prepared_seeding': (None if mesh.seeding is None else {
                'divisions': dict(mesh.seeding.divisions), 'classes': dict(mesh.seeding.classes)}),
            'source_seeding_size_field': (None if mesh.seeding is None or mesh.seeding.size_field is None else {
                'kind': 'uniform', 'target_size': mesh.seeding.size_field.target_size,
                'source_model_id': str(source.model_id)}),
            'edge_parameter_maps': [(current.id, original.id, str(lo), str(hi))
                                    for current, original, lo, hi in receipt.edge_parameter_maps]}
        preparation['closure_source_transfer'] = {
            'source_model_id': str(source.model_id), 'source_revision': source.revision,
            'working_model_id': str(work.model_id), 'working_revision': work.revision,
            'prepared_model_id': str(prepared.model_id), 'prepared_revision': prepared.revision}
        candidates = curvilinear_reference_candidates(mesh)
        original_registry = curvilinear_reference_registry(mesh)
        generation_registry = mesh.boundary_registry
        for name, value in tables.items():
            setattr(mesh, name, value)
        mesh.geometry_model_id, mesh.geometry_revision = source.model_id, source.revision
        mesh.boundary_registry = registry
        mesh.structural_preparation = preparation
        key = id(mesh)

        def cleanup(reference):
            entry = _IDENTITY_PUBLICATIONS.get(key)
            if entry is not None and entry[0] is reference:
                _IDENTITY_PUBLICATIONS.pop(key, None)

        _IDENTITY_PUBLICATIONS[key] = (ref(mesh, cleanup), source, prepared, receipt, candidates,
            original_registry, generation_registry, registry, boundary_registry_pin(registry),
            _publication_pin(mesh)) + ((station_transport,) if station_transport is not None else ())
        _validate_identity_source_publication(source, prepared, mesh, registry)
        preparation['qualified_s3'] = _qualify_identity_source_s3(source, prepared, mesh)
        mesh.structural_preparation = preparation
        check('closure source publication complete')
        if qualification is not None:
            _closure_publication_qualification(source, closure, qualification, check_source=True)
            from anygeometry import validate_prepared_material_source_transfer_binding
            for working_owner, working_receipt in working_transfers:
                validate_prepared_material_source_transfer_binding(
                    working_owner, prepared, working_receipt,
                    cell_receipts=tuple(item[3] for item in candidates))
        _validate_identity_source_publication(source, prepared, mesh, registry)
        return mesh
    except BaseException:
        snapshot.restore()
        if old_references is None:
            _CURVILINEAR_MESH_REFERENCES.pop(id(mesh), None)
        else:
            _CURVILINEAR_MESH_REFERENCES[id(mesh)] = old_references
        if old_publication is None:
            _IDENTITY_PUBLICATIONS.pop(id(mesh), None)
        else:
            _IDENTITY_PUBLICATIONS[id(mesh)] = old_publication
        raise
