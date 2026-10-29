"""Actual-engine source station aliases; definitions AUTHORED NOT RUN.

Owner receipts remain mandatory. A nominal float never changes ancestry/XYZ.
"""
from copy import copy
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anymesher import (qualify_model_closure_source_publication,
    publish_model_closure_mesh_to_source, validate_material_source_transfer_for_mesh)
from anymesher.errors import MeshError
from anymesher._source_boundary_publication import source_boundaries
from anymesher._publication_snapshot import PublicationSnapshot
import anymesher.material_source_transfer as publication


@pytest.fixture(scope='module')
def flat_split_mesh():
    model = owner.GeometryModel()
    surface = NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))))
    uv = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    vertices = [model.add_point(*surface.evaluate(*p)) for p in uv]
    edges = [model.add_curve(vertices[i],vertices[(i+1)%4],CurveOnSurface(surface,
        NurbsCurve(1,(0.,0.,1.,1.),(p,uv[(i+1)%4])))) for i,p in enumerate(uv)]
    face = model.add_face(edges,surface=surface)
    model.add_sheet((face,),name='flat source')
    before = owner.to_dict(model)
    closure = owner.extract_model_closure(model,tuple(model.handle('sheet',s) for s in model.sheets))
    token = qualify_model_closure_source_publication(model,closure)
    work = closure.working_model
    plane = owner.Plane((.6,0.,0.),(0.,1.,0.),(0.,0.,1.))
    plan = owner.plan_plane_split(work,list(work.faces),plane)
    application = owner.apply_plane_split(work,plan,policy='imprint')
    assert len(work.faces) == 2 and application.trace_edges
    from anymesher.recovery import select_curvilinear_recovery_policy,generate_automatic_mesh_result
    policy,selection = select_curvilinear_recovery_policy(work,requested_strategy='quad_first',
        order='linear',certification_mode='interactive',geometry_approximation_error=.0025,
        explicit_controls=False)
    result = generate_automatic_mesh_result(work,automation=policy,
        curvilinear_recovery_selection=selection,target_size=.25,strategy='quad_first',
        order='linear',certification_mode='interactive',geometry_approximation_error=.0025,
        qualified_s3=True,structural_preparation=False,mutation_policy='working_copy',
        native_backend='python',recombine=True)
    mesh = result.mesh
    assert result.status == 'ready' and mesh.structural_preparation['qualified_s3']['status'] == 'ADMITTED'
    prepared = mesh.boundary_registry.view.source
    assert owner.to_dict(model) == before
    return model,closure,token,prepared,mesh,before


@pytest.fixture
def alias_context(flat_split_mesh):
    source,closure,token,prepared,mesh,before = flat_split_mesh
    old = publication._IDENTITY_PUBLICATIONS.get(id(mesh))
    snapshot = PublicationSnapshot(mesh,old)
    transfer = validate_material_source_transfer_for_mesh(source,prepared,mesh)
    aliases = owner.query_prepared_source_station_transport(source,prepared,transfer)
    assert aliases.rows and any(Q(row.nominal_parameter) != row.source_parameter for row in aliases.rows)
    vertices = {target.id:original.id for target,original in closure.work_to_source.items() if target.kind=='vertex'}
    try:
        yield SimpleNamespace(source=source,closure=closure,token=token,prepared=prepared,
            mesh=mesh,transfer=transfer,aliases=aliases,vertices=vertices,before=before)
    finally:
        snapshot.restore()
        if old is None:
            publication._IDENTITY_PUBLICATIONS.pop(id(mesh),None)
        else:
            publication._IDENTITY_PUBLICATIONS[id(mesh)] = old


@pytest.mark.parametrize('passive', (False,True))
def test_public_nonbinary_alias_publication_keeps_literal_cells_and_source_s3(alias_context,monkeypatch,passive):
    c = alias_context
    queried = []
    callback = owner.CancellationSignal() if passive else None
    actual = owner.query_prepared_source_station_transport
    def observe(*args,**kwargs):
        assert kwargs['cancellation_check'] is callback
        result = actual(*args,**kwargs)
        queried.append(result)
        return result
    monkeypatch.setattr(owner,'query_prepared_source_station_transport',observe)
    c.mesh.structural_preparation['station_transport'] = {'nominal_parameter':'forged metadata'}
    literal = dict(c.mesh.nodes),dict(c.mesh.tris),dict(c.mesh.quads)
    publish_model_closure_mesh_to_source(c.source,c.closure,c.mesh,qualification=c.token,
        cancellation_check=callback)
    assert len(queried) == 1
    entry = publication._IDENTITY_PUBLICATIONS[id(c.mesh)]
    assert entry[10] is queried[0]
    publication._validate_identity_source_publication(c.source,c.prepared,c.mesh,c.mesh.boundary_registry)
    assert c.mesh.geometry_model_id == c.source.model_id
    assert c.mesh.structural_preparation['qualified_s3']['status'] == 'ADMITTED'
    provenance = c.mesh.structural_preparation['prepared_boundary_provenance']
    old_seed = provenance['prepared_seeding']
    from collections import Counter
    parent_children = {}
    for child,parent,_a,_b in c.transfer.edge_parameter_maps:
        parent_children.setdefault(parent.id,[]).append(child.id)
    signatures = {parent:tuple(sorted(Counter(old_seed['classes'][edge]
        for edge in children).items())) for parent,children in parent_children.items()}
    assert c.mesh.seeding.divisions == {parent:sum(old_seed['divisions'][edge]
        for edge in children) for parent,children in parent_children.items()}
    for first in signatures:
        for second in signatures:
            assert (c.mesh.seeding.classes[first] == c.mesh.seeding.classes[second]) == (signatures[first] == signatures[second])
    assert c.mesh.seeding.size_field.is_uniform
    assert provenance['source_seeding_size_field']['target_size'] == .25
    assert (dict(c.mesh.nodes),dict(c.mesh.tris),dict(c.mesh.quads)) == literal
    for row in queried[0].rows:
        expected = next(e for e in c.mesh.boundary_registry.entries(row.source_edge.id)
                        if e.key.parameter == row.nominal_parameter)
        original = next(e for e in entry[6].entries(row.prepared_edge.id)
                        if Q(e.key.parameter) == row.prepared_parameter)
        assert expected.node_id == original.node_id
        assert tuple(expected.point) == tuple(original.point) == row.xyz
    assert owner.to_dict(c.source) == c.before


@pytest.mark.parametrize('attack',('copy','metadata','raw_alias'))
def test_alias_consumer_requires_live_unchanged_owner_result(alias_context,attack):
    c = alias_context
    result = c.aliases
    if attack == 'copy':
        result = copy(result)
    elif attack == 'metadata':
        result = SimpleNamespace(rows=c.aliases.rows)
    else:
        object.__setattr__(result.rows[0],'nominal_parameter',.875)
    with pytest.raises((MeshError,owner.GeometryError)):
        source_boundaries(c.source,c.prepared,c.mesh,c.transfer,c.vertices,station_transport=result)


@pytest.mark.parametrize('attack',('node_identity','coordinate'))
def test_alias_consumer_rejects_existing_station_identity_or_xyz_change(alias_context,attack):
    c = alias_context
    row = next(row for row in c.aliases.rows if Q(row.nominal_parameter) != row.source_parameter)
    entry = next(e for e in c.mesh.boundary_registry.entries(row.prepared_edge.id)
                 if Q(e.key.parameter) == row.prepared_parameter)
    if attack == 'coordinate':
        point = tuple(float(x)+.001 for x in entry.point)
        object.__setattr__(entry,'point',point)
        c.mesh.nodes[entry.node_id] = point
    else:
        first,second = next((a,b) for a in c.aliases.rows for b in c.aliases.rows
            if a.source_edge == b.source_edge and a.source_parameter == b.source_parameter
            and a.prepared_edge != b.prepared_edge)
        other = next(e for e in c.mesh.boundary_registry.entries(second.prepared_edge.id)
                     if Q(e.key.parameter) == second.prepared_parameter)
        node = max(c.mesh.nodes)+1
        c.mesh.nodes[node] = tuple(other.point)
        object.__setattr__(other,'node_id',node)
    with pytest.raises(MeshError,match='literal|global node|correspondence|identit'):
        source_boundaries(c.source,c.prepared,c.mesh,c.transfer,c.vertices,station_transport=c.aliases)


def test_alias_publication_query_callback_baseexception_restores_owned_mesh(alias_context):
    c = alias_context
    before = publication._publication_pin(c.mesh)
    nodes,registry = c.mesh.nodes,c.mesh.boundary_registry
    class Cancelled(BaseException):
        pass
    failure = Cancelled()
    hit = []
    def callback(phase):
        if 'source station transport' in phase:
            hit.append(phase)
            c.mesh.nodes[next(iter(c.mesh.nodes))] = (9.,9.,9.)
            raise failure
        return False
    with pytest.raises(Cancelled) as caught:
        publish_model_closure_mesh_to_source(c.source,c.closure,c.mesh,
            qualification=c.token,cancellation_check=callback)
    assert hit and caught.value is failure
    assert c.mesh.nodes is nodes and c.mesh.boundary_registry is registry
    assert publication._publication_pin(c.mesh) == before
    assert owner.to_dict(c.source) == c.before


def test_alias_consumer_refuses_unsupported_offset_reference(alias_context):
    c = alias_context
    edge = c.aliases.rows[0].prepared_edge.id
    c.mesh.offset_nodes_of_edge[edge] = list(c.mesh.nodes_of_edge[edge])
    with pytest.raises(MeshError,match='offset or member'):
        source_boundaries(c.source,c.prepared,c.mesh,c.transfer,c.vertices,station_transport=c.aliases)


def test_exact_binary_publication_does_not_query_aliases_or_change_legacy_entry(monkeypatch):
    from test_model_closure_source_publication import qualified_closure_publication_operands
    source,closure,prepared,mesh,token = qualified_closure_publication_operands()
    def unexpected(*_args,**_kwargs):
        raise AssertionError('exact-binary legacy route queried station aliases')
    monkeypatch.setattr(owner,'query_prepared_source_station_transport',unexpected)
    publish_model_closure_mesh_to_source(source,closure,mesh,qualification=token)
    assert len(publication._IDENTITY_PUBLICATIONS[id(mesh)]) == 10
    publication._validate_identity_source_publication(source,prepared,mesh,mesh.boundary_registry)
    assert mesh.structural_preparation['qualified_s3']['status'] == 'ADMITTED'


def test_alias_publication_passive_cancellation_after_query_restores_owned_state(alias_context,monkeypatch):
    c = alias_context
    signal = owner.CancellationSignal()
    actual = owner.query_prepared_source_station_transport
    before = publication._publication_pin(c.mesh)
    previous = publication._IDENTITY_PUBLICATIONS.get(id(c.mesh))
    registry,nodes = c.mesh.boundary_registry,c.mesh.nodes
    def cancel_after(*args,**kwargs):
        assert kwargs['cancellation_check'] is signal
        result = actual(*args,**kwargs)
        signal.cancel()
        return result
    monkeypatch.setattr(owner,'query_prepared_source_station_transport',cancel_after)
    with pytest.raises((MeshError,owner.GeometryError),match='cancelled'):
        publish_model_closure_mesh_to_source(c.source,c.closure,c.mesh,
            qualification=c.token,cancellation_check=signal)
    assert signal.cancelled
    assert c.mesh.boundary_registry is registry and c.mesh.nodes is nodes
    assert publication._publication_pin(c.mesh) == before
    assert publication._IDENTITY_PUBLICATIONS.get(id(c.mesh)) is previous


def test_alias_result_cannot_authorize_a_copied_mesh_generation_context(alias_context):
    c = alias_context
    pin = publication._publication_pin(c.mesh)
    prior = publication._IDENTITY_PUBLICATIONS.get(id(c.mesh))
    clone = copy(c.mesh)  # supported shallow copy; proof authority is mesh identity
    assert clone is not c.mesh and clone.nodes is c.mesh.nodes
    assert clone.boundary_registry is c.mesh.boundary_registry
    with pytest.raises((MeshError,owner.GeometryError)):
        source_boundaries(c.source,c.prepared,clone,c.transfer,c.vertices,
            station_transport=c.aliases)
    assert publication._publication_pin(c.mesh) == pin
    assert publication._IDENTITY_PUBLICATIONS.get(id(c.mesh)) is prior
    assert owner.to_dict(c.source) == c.before

def test_actual_publisher_rejects_unique_station_node_alias_at_generation_gate(alias_context,monkeypatch):
    """Same XYZ cannot grant a new ID to an already generation-owned station."""
    from collections import Counter
    import anymesher._source_boundary_publication as boundaries
    c = alias_context
    counts = Counter((row.source_edge.id,row.source_parameter) for row in c.aliases.rows)
    station = next(row for row in c.aliases.rows
        if counts[(row.source_edge.id,row.source_parameter)] == 1
        and 0 < row.prepared_parameter < 1)
    entry = next(row for row in c.mesh.boundary_registry.entries(station.prepared_edge.id)
                 if Q(row.key.parameter) == station.prepared_parameter)
    nodes,registry = c.mesh.nodes,c.mesh.boundary_registry
    old_node = entry.node_id
    same_xyz = tuple(entry.point)
    new_node = max(c.mesh.nodes)+1
    c.mesh.nodes[new_node] = same_xyz
    object.__setattr__(entry,'node_id',new_node)
    # The caller's raw alteration is the entry baseline. Refusal must retain
    # it without publishing/remapping anything; fixture teardown restores the
    # pre-attack aliases for other tests using this genuine generation.
    attacked = publication._publication_pin(c.mesh)
    previous = publication._IDENTITY_PUBLICATIONS.get(id(c.mesh))
    def forbidden(*_args,**_kwargs):
        raise AssertionError('changed station reached alias/remapping instead of generation refusal')
    monkeypatch.setattr(boundaries,'source_boundaries',forbidden)
    monkeypatch.setattr(owner,'query_prepared_source_station_transport',forbidden)
    with pytest.raises(MeshError,match='generation authority|cells, stations or associations changed'):
        publish_model_closure_mesh_to_source(c.source,c.closure,c.mesh,qualification=c.token)
    assert publication._publication_pin(c.mesh) == attacked
    assert publication._IDENTITY_PUBLICATIONS.get(id(c.mesh)) is previous
    assert c.mesh.nodes is nodes and c.mesh.boundary_registry is registry
    assert entry.node_id == new_node and tuple(entry.point) == same_xyz
    assert tuple(c.mesh.nodes[old_node]) == tuple(c.mesh.nodes[new_node]) == same_xyz
    assert owner.to_dict(c.source) == c.before


@pytest.mark.parametrize('attack',('target_size','contents'))
@pytest.mark.parametrize('failure_kind',('baseexception','cancelled'))
def test_actual_publisher_size_field_callback_rollback_preserves_prior_authority(
        alias_context,attack,failure_kind):
    c = alias_context
    # Establish genuine prior source publication before snapshotting the attack.
    publish_model_closure_mesh_to_source(c.source,c.closure,c.mesh,qualification=c.token)
    previous = publication._IDENTITY_PUBLICATIONS.get(id(c.mesh))
    assert previous is not None
    publication._validate_identity_source_publication(c.source,c.prepared,c.mesh,c.mesh.boundary_registry)
    field = c.mesh.seeding.size_field
    sizes = field.edge_sizes
    old_target,old_sizes = field.target_size,dict(sizes)
    seed,registry,nodes = c.mesh.seeding,c.mesh.boundary_registry,c.mesh.nodes
    before = publication._publication_pin(c.mesh)
    class Interrupted(BaseException):
        pass
    failure = Interrupted() if failure_kind == 'baseexception' else owner.GeometryError('field callback cancelled')
    hit = []
    def callback(phase):
        hit.append(phase)
        if attack == 'target_size':
            field.target_size = old_target/2
        else:
            sizes[next(iter(c.source.edges))] = old_target/2
        raise failure
    with pytest.raises(type(failure)) as caught:
        publish_model_closure_mesh_to_source(c.source,c.closure,c.mesh,
            qualification=c.token,cancellation_check=callback)
    assert hit and caught.value is failure
    assert c.mesh.seeding is seed and c.mesh.seeding.size_field is field
    assert field.edge_sizes is sizes and sizes == old_sizes and field.target_size == old_target
    assert c.mesh.boundary_registry is registry and c.mesh.nodes is nodes
    assert publication._publication_pin(c.mesh) == before
    assert publication._IDENTITY_PUBLICATIONS.get(id(c.mesh)) is previous
    publication._validate_identity_source_publication(c.source,c.prepared,c.mesh,registry)
    assert c.mesh.structural_preparation['qualified_s3']['status'] == 'ADMITTED'
    assert owner.to_dict(c.source) == c.before
