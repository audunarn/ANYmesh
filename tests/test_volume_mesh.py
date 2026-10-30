"""Controlled provider fixtures: these are not real-mesher qualification."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest

from anygeometry.identity import EntityHandle
from anymesher.volume import (
    VolumeMesh, InvalidVolumeMesh, UnsupportedVolumeGeometry,
    VolumeBackendUnavailable, VolumeMeshingCancelled,
    generate_volume_mesh, validate_volume_mesh,
)
from anymesher._volume_validation import (validate_source, validate_cells,
    validate_geometry, _check_pair, _tet_planes)


def box(height=1., *, transform=None):
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, height], [1, 0, height], [1, 1, height], [0, 1, height]], dtype=float)
    if transform is not None:
        coordinates = coordinates @ transform.T + np.array([3., -2., 5.])
    model_id = '71b52353-5ac5-4743-95b4-fc32c2347baf'
    refs = tuple(EntityHandle(model_id, 'face', i + 1) for i in range(6))
    loops = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    snapshot = SimpleNamespace(schema_version=1, model_id=model_id, revision=7,
        region_id='solid-1', units='m', geometry_class='convex_normal_prism',
        coordinates=coordinates, face_loops=loops, face_refs=refs,
        face_groups={'start': (refs[0],), 'end': (refs[1],),
                     'loaded': (refs[1], refs[2]), 'side_0': (refs[2],)},
        volume=height, bounds=np.array([coordinates.min(axis=0), coordinates.max(axis=0)]),
        source_signature='a' * 64, provenance={'fixture': 'analytic unit square normal prism'})
    cells = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                      [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    return snapshot, cells


def make_mesh(snapshot, cells, delta=1e-7, **kwargs):
    source = validate_source(snapshot, delta)
    facets = validate_cells(snapshot.coordinates, cells)
    owners, _ = validate_geometry(snapshot.coordinates, cells, facets, source)
    ids = np.arange(300, 300 + len(facets))
    groups = {name: ids[[h in handles for h in owners]] for name, handles in snapshot.face_groups.items()}
    metadata = dict(source_model_id=snapshot.model_id, source_revision=snapshot.revision,
                    source_region_id=snapshot.region_id, source_signature=snapshot.source_signature)
    metadata.update(kwargs.pop('provenance', {}))
    cell_groups = {snapshot.region_id: np.arange(200, 206), **kwargs.pop('cell_groups', {})}
    return VolumeMesh(snapshot.coordinates, cells, node_ids=np.arange(100, 108),
        cell_ids=np.arange(200, 206), boundary_facets=facets, boundary_ids=ids,
        boundary_groups=groups, face_owners=dict(zip(map(int, ids), owners)), provenance=metadata, cell_groups=cell_groups, **kwargs)


def test_generated_region_group_uses_external_cell_ids(monkeypatch):
    snapshot, cells = box()
    fake_provider(monkeypatch, snapshot.coordinates, cells)
    mesh = generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7)
    np.testing.assert_array_equal(mesh.cell_groups[snapshot.region_id], mesh.cell_ids)
    fields = ('coordinates','cells','node_ids','cell_ids','boundary_facets','boundary_ids',
              'node_groups','cell_groups','boundary_groups','units','geometry_order','provenance','face_owners')
    raw = SimpleNamespace(**{name: getattr(mesh, name) for name in fields})
    raw.cell_groups = {}
    with pytest.raises(InvalidVolumeMesh, match='region'):
        validate_volume_mesh(raw, snapshot, boundary_tolerance=1e-7)


def fake_provider(monkeypatch, coordinates, cells):
    calls = []
    def run(source, edge, timeout, cancel):
        calls.append((source, edge, timeout))
        return coordinates.copy(), cells.copy(), {'backend': 'controlled_fixture'}
    monkeypatch.setattr('anymesher._volume_process.run_provider', run)
    return calls


def test_immutable_complete_boundary_external_ids_and_overlapping_groups():
    snapshot, cells = box()
    mesh = make_mesh(snapshot, cells, node_groups={'fixed': [100, 101]}, cell_groups={'body': [200, 205]}, provenance={'nested': [{'value': 2}]})
    assert len(mesh.boundary_facets) == 12
    assert mesh.boundary_groups['end'].size == 2
    assert set(mesh.boundary_groups['end']) <= set(mesh.boundary_groups['loaded'])
    assert len(mesh.face_owners) == 12
    assert validate_volume_mesh(mesh, snapshot, boundary_tolerance=1e-7)['volume'] == pytest.approx(1.)
    snapshot.coordinates[0] += .1
    for array in (mesh.coordinates, mesh.cells, mesh.node_ids, mesh.cell_ids,
                  mesh.boundary_facets, mesh.boundary_ids, mesh.node_groups['fixed'],
                  mesh.cell_groups['body'], mesh.boundary_groups['loaded']):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises((FrozenInstanceError, AttributeError)):
        mesh.units = 'mm'
    with pytest.raises(TypeError):
        mesh.provenance['nested'][0]['value'] = 3
    with pytest.raises(TypeError):
        mesh.face_owners[300] = None
    assert np.array_equal(mesh.coordinates[0], [0., 0., 0.])


@pytest.mark.parametrize('height', [1., .001, 1e-6])
def test_box_and_thin_prisms_generate_from_controlled_raw_arrays(monkeypatch, height):
    theta = .43
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    snapshot, cells = box(height, transform=rotation)
    delta = height * 1e-4
    calls = fake_provider(monkeypatch, snapshot.coordinates, cells[:, [0, 2, 1, 3]])
    mesh = generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=delta)
    assert mesh.provenance['orientation_swaps'] == 6
    assert mesh.provenance['source_signature'] == snapshot.source_signature
    assert calls[0][1:] == (.2, 120.)
    assert validate_volume_mesh(mesh, snapshot, boundary_tolerance=delta)['volume'] == pytest.approx(height)


@pytest.mark.parametrize('field,value,exc', [
    ('units', 'mm', UnsupportedVolumeGeometry),
    ('schema_version', 2, UnsupportedVolumeGeometry),
    ('geometry_class', 'concave', UnsupportedVolumeGeometry),
    ('bounds', [[0, 0, 0], [2, 2, 2]], InvalidVolumeMesh),
    ('volume', 3., InvalidVolumeMesh),
    ('source_signature', 'x', InvalidVolumeMesh),
    ('revision', True, InvalidVolumeMesh),
    ('model_id', 'bad', InvalidVolumeMesh),
])
def test_source_refusal_before_provider(monkeypatch, field, value, exc):
    snapshot, cells = box()
    calls = fake_provider(monkeypatch, snapshot.coordinates, cells)
    setattr(snapshot, field, value)
    with pytest.raises(exc):
        generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7)
    assert not calls


def test_source_rejects_wrong_owner_unknown_group_open_nonplanar_and_oblique():
    for mutation in ('owner', 'group', 'open', 'nonplanar', 'oblique', 'reversed'):
        snapshot, _ = box()
        if mutation == 'owner':
            snapshot.face_refs = (*snapshot.face_refs[:-1], EntityHandle('26ebcf4d-d88e-4cd1-802c-e5cbd76bb3df', 'face', 10))
        elif mutation == 'group':
            snapshot.face_groups['bad'] = (EntityHandle(snapshot.model_id, 'face', 99),)
        elif mutation == 'open':
            snapshot.face_loops = snapshot.face_loops[:-1]
            snapshot.face_refs = snapshot.face_refs[:-1]
            snapshot.face_groups = {}
        elif mutation in ('nonplanar', 'oblique'):
            snapshot.coordinates[4 if mutation == 'nonplanar' else slice(4, 8), 0] += .1
            if mutation == 'oblique':
                snapshot.coordinates[4:, 1] += .2
            snapshot.bounds = np.array([snapshot.coordinates.min(axis=0), snapshot.coordinates.max(axis=0)])
        else:
            snapshot.face_loops = (snapshot.face_loops[0][::-1], *snapshot.face_loops[1:])
        with pytest.raises((InvalidVolumeMesh, UnsupportedVolumeGeometry)):
            validate_source(snapshot, 1e-7)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, None, 'x', [], 1+2j])
def test_invalid_sizing_and_tolerance(monkeypatch, value):
    snapshot, cells = box()
    calls = fake_provider(monkeypatch, snapshot.coordinates, cells)
    for name in ('target_edge_length', 'boundary_tolerance', 'timeout'):
        args = dict(target_edge_length=.2, boundary_tolerance=1e-7)
        args[name] = value
        with pytest.raises(InvalidVolumeMesh):
            generate_volume_mesh(snapshot, **args)
    assert not calls


def test_tolerance_cannot_hide_thin_features():
    snapshot, _ = box(.001)
    with pytest.raises(InvalidVolumeMesh, match='feature separation'):
        validate_source(snapshot, .0002)


def test_explicit_unavailable_backend_and_precancel(monkeypatch):
    snapshot, cells = box()
    calls = fake_provider(monkeypatch, snapshot.coordinates, cells)
    with pytest.raises(VolumeBackendUnavailable):
        generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7, backend='other')
    with pytest.raises(VolumeMeshingCancelled):
        generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7, cancel=lambda: True)
    assert not calls


@pytest.mark.parametrize('mutation', ['negative', 'degenerate', 'duplicate_cell', 'duplicate_node', 'unused', 'unknown', 'repeated', 'float', 'nan', 'disconnected'])
def test_independent_cell_validation_refuses_corruption(mutation):
    snapshot, t = box()
    x = snapshot.coordinates.copy()
    if mutation == 'negative': t[0, [1, 2]] = t[0, [2, 1]]
    elif mutation == 'degenerate': x[6] = x[0]
    elif mutation == 'duplicate_cell': t = np.vstack((t, t[0]))
    elif mutation == 'duplicate_node': x[7] = x[3]
    elif mutation == 'unused': x = np.vstack((x, [2, 2, 2]))
    elif mutation == 'unknown': t[0, 0] = 100
    elif mutation == 'repeated': t[0, 1] = t[0, 0]
    elif mutation == 'float': t = t.astype(float)
    elif mutation == 'nan': x[0, 0] = float('nan')
    else:
        x = np.vstack((x, x + 3))
        t = np.vstack((t, t + 8))
    with pytest.raises(InvalidVolumeMesh):
        validate_cells(x, t)


def test_overlap_and_nonconforming_contact_predicates_against_simple_tetrahedra():
    p = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    q = p + .1
    with pytest.raises(InvalidVolumeMesh, match='overlap'):
        _check_pair(p, q, np.empty((0, 3)), _tet_planes(p), _tet_planes(q))
    # Geometric point contact, but different row topology, must also be refused.
    q = p + [1., 0, 0]
    with pytest.raises(InvalidVolumeMesh, match='nonconforming'):
        _check_pair(p, q, np.empty((0, 3)), _tet_planes(p), _tet_planes(q))
    _check_pair(p, q, p[[1]], _tet_planes(p), _tet_planes(q))
    _check_pair(p, p + 2, np.empty((0, 3)), _tet_planes(p), _tet_planes(p + 2))


def test_boundary_is_complete_unique_and_outward():
    snapshot, cells = box()
    mesh = make_mesh(snapshot, cells)
    for mutation in ('missing', 'duplicate', 'reversed', 'owner', 'groups'):
        raw = SimpleNamespace(**{name: getattr(mesh, name) for name in mesh.__dataclass_fields__})
        if mutation == 'missing': raw.boundary_facets = mesh.boundary_facets[:-1]
        elif mutation == 'duplicate': raw.boundary_facets = np.vstack((mesh.boundary_facets[:-1], mesh.boundary_facets[0]))
        elif mutation == 'reversed':
            raw.boundary_facets = mesh.boundary_facets.copy()
            raw.boundary_facets[0] = raw.boundary_facets[0][::-1]
        elif mutation == 'owner':
            raw.face_owners = dict(mesh.face_owners)
            raw.face_owners[300] = snapshot.face_refs[1]
        else: raw.boundary_groups = {'end': mesh.boundary_groups['end']}
        with pytest.raises(InvalidVolumeMesh):
            validate_volume_mesh(raw, snapshot, boundary_tolerance=1e-7)


@pytest.mark.parametrize('mutation', ['outside', 'missing_corner', 'inset', 'wrong_plane'])
def test_geometric_validation_refuses_boundary_damage(monkeypatch, mutation):
    snapshot, cells = box()
    x = snapshot.coordinates.copy()
    if mutation == 'outside': x[6, 0] += .01
    elif mutation == 'missing_corner': x[6] -= [.01, .01, 0]
    elif mutation == 'inset': x = .99 * x + .005
    else: x[:, 2] += .01
    fake_provider(monkeypatch, x, cells)
    with pytest.raises(InvalidVolumeMesh):
        generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7)


def test_public_import_is_lazy_for_optional_backend_and_consumer():
    result = subprocess.run([sys.executable, '-c',
        """
import sys
import anymesher
from anymesher import approximation_policy, boundary_orientation, material_source_transfer, volume

# Main's shell/material API and Workflow2's solid API must coexist at the public
# package boundary without loading optional providers or solver consumers.
for module, names in (
    (volume, ('VolumeMesh', 'generate_volume_mesh', 'validate_volume_mesh')),
    (approximation_policy, ('GeometryApproximationPolicy', 'resolve_geometry_approximation')),
    (boundary_orientation, ('orient_displacement_boundary', 'generate_displacement_boundary_mesh_result')),
    (material_source_transfer, ('validate_material_source_transfer_for_mesh', 'publish_model_closure_mesh_to_source')),
):
    for name in names:
        assert name in anymesher.__all__, name
        assert getattr(anymesher, name) is getattr(module, name), name
assert not {'pytetwild', 'pyvista', 'vtk', 'anysolid', 'anysolver', 'ngsolve'} & set(sys.modules)
"""],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_atomic_cancellation_after_provider(monkeypatch):
    snapshot, cells = box()
    flag = [False]
    def run(*args):
        flag[0] = True
        return snapshot.coordinates, cells, {}
    monkeypatch.setattr('anymesher._volume_process.run_provider', run)
    with pytest.raises(VolumeMeshingCancelled):
        generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7, cancel=lambda: flag[0])


def test_signed_extrusion_and_triangular_normal_prism(monkeypatch):
    snapshot, cells = box(-.001)
    snapshot.volume = .001
    snapshot.face_loops = tuple(loop[::-1] for loop in snapshot.face_loops)
    fake_provider(monkeypatch, snapshot.coordinates, cells)
    mesh = generate_volume_mesh(snapshot, target_edge_length=.1, boundary_tolerance=1e-8)
    assert validate_volume_mesh(mesh, snapshot, boundary_tolerance=1e-8)['volume'] == pytest.approx(.001)
    snapshot, _ = box()
    snapshot.coordinates = np.array([[0.,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1]])
    snapshot.face_loops = ((0,2,1), (3,4,5), (0,1,4,3), (1,2,5,4), (2,0,3,5))
    snapshot.face_refs = snapshot.face_refs[:5]
    snapshot.face_groups = {'start': snapshot.face_refs[:1], 'end': snapshot.face_refs[1:2]}
    snapshot.volume = .5
    cells = np.array([[0,1,2,3], [1,2,3,4], [2,3,4,5]])
    fake_provider(monkeypatch, snapshot.coordinates, cells)
    mesh = generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7)
    assert validate_volume_mesh(mesh, snapshot, boundary_tolerance=1e-7)['volume'] == pytest.approx(.5)


@pytest.mark.parametrize('key,value', [('revision', 8), ('source_signature', 'b'*64), ('region_id', 'other')])
def test_stale_source_refused(key, value):
    snapshot, cells = box()
    mesh = make_mesh(snapshot, cells)
    setattr(snapshot, key, value)
    with pytest.raises(InvalidVolumeMesh, match='provenance'):
        validate_volume_mesh(mesh, snapshot, boundary_tolerance=1e-7)


def test_mutated_protocol_ids_units_and_owner_assignment_refused():
    snapshot, cells = box()
    mesh = make_mesh(snapshot, cells)
    for name, value in [('node_ids', [1]*8), ('cell_ids', [1]*6), ('units', 'mm'),
                        ('geometry_order', 2), ('face_owners', {}),
                        ('node_groups', {'bad': [99999]}), ('cell_groups', {'bad': [99999]})]:
        raw = SimpleNamespace(**{field: getattr(mesh, field) for field in mesh.__dataclass_fields__})
        setattr(raw, name, value)
        with pytest.raises(InvalidVolumeMesh):
            validate_volume_mesh(raw, snapshot, boundary_tolerance=1e-7)


def test_constructor_refuses_partial_or_inward_boundary_and_unknown_groups():
    snapshot, cells = box()
    mesh = make_mesh(snapshot, cells)
    with pytest.raises(InvalidVolumeMesh):
        VolumeMesh(snapshot.coordinates, cells, boundary_facets=mesh.boundary_facets[:-1], face_owners=mesh.face_owners)
    with pytest.raises(InvalidVolumeMesh):
        VolumeMesh(snapshot.coordinates, cells, boundary_facets=mesh.boundary_facets[:, ::-1], face_owners=mesh.face_owners)
    with pytest.raises(InvalidVolumeMesh):
        VolumeMesh(snapshot.coordinates, cells, face_owners={})


def test_exact_contact_oracle_detects_invalid_nonconforming_shared_edge():
    from anymesher._volume_validation import _exact_contact
    p = np.array([[0.,0,0], [1,0,0], [0,1,0], [0,0,1]])
    # Full overlapping tetrahedra cannot be reinterpreted as sharing only an edge.
    with pytest.raises(InvalidVolumeMesh, match='exact predicate'):
        _exact_contact(p, p, p[:2])


def test_nonmanifold_and_same_side_cells_refused():
    x = np.array([[0.,0,0], [1,0,0], [0,1,0], [0,0,1], [.1,.1,2], [.2,.2,3]])
    for cells in (np.array([[0,1,2,3], [0,1,2,4]]), np.array([[0,1,2,3], [0,1,2,4], [0,1,2,5]])):
        with pytest.raises(InvalidVolumeMesh):
            validate_cells(x[:5] if len(cells) == 2 else x, cells)


def test_explicit_boundary_tolerance_admits_bounded_inset(monkeypatch):
    snapshot, cells = box()
    delta = 1e-5
    x = snapshot.coordinates * (1 - delta / 5) + delta / 10
    fake_provider(monkeypatch, x, cells)
    mesh = generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=delta)
    assert validate_volume_mesh(mesh, snapshot, boundary_tolerance=delta)['volume'] < 1.


def test_validation_cancellation_is_atomic():
    snapshot, cells = box()
    with pytest.raises(VolumeMeshingCancelled):
        validate_cells(snapshot.coordinates, cells, cancel=lambda: True)


def test_complex_or_ragged_raw_arrays_are_typed_refusals(monkeypatch):
    snapshot, cells = box()
    fake_provider(monkeypatch, snapshot.coordinates.astype(complex) + 1j, cells)
    with pytest.raises(InvalidVolumeMesh, match='real numerical'):
        generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7)
    with pytest.raises(InvalidVolumeMesh, match='rectangular integer'):
        validate_cells(snapshot.coordinates, [[0,1,2,6], [0,2,3]])


def test_canonical_rows_ids_and_raw_correspondence_survive_provider_permutation(monkeypatch):
    snapshot, cells = box()
    fake_provider(monkeypatch, snapshot.coordinates, cells)
    first = generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7)
    order = np.array([7, 2, 1, 4, 0, 6, 5, 3])
    inverse = np.argsort(order)
    cell_order = np.array([4, 1, 3, 2, 5, 0])
    raw_x, raw_t = snapshot.coordinates[order], inverse[cells[cell_order]]
    fake_provider(monkeypatch, raw_x, raw_t)
    second = generate_volume_mesh(snapshot, target_edge_length=.2, boundary_tolerance=1e-7)
    for name in ('coordinates', 'cells', 'node_ids', 'cell_ids', 'boundary_facets', 'boundary_ids'):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    node_map = np.array(second.provenance['raw_node_row_to_row'])
    cell_map = np.array(second.provenance['raw_cell_row_to_row'])
    assert np.array_equal(raw_x, second.coordinates[node_map])
    assert np.array_equal(np.sort(node_map[raw_t], axis=1), np.sort(second.cells[cell_map], axis=1))
