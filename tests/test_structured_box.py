"""Structured rectangular-box Tet4 lattice: generation, admission and consumers.

The consumer checks below (volume, boundary coverage, outward normals, rotation,
persistence) are computed here from raw arrays, independently of ANYmesher's
admission code, and are not provider qualification.
"""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import anymesher
from anygeometry import GeometryModel
from anygeometry.identity import EntityHandle
from anymesher import (InvalidVolumeMesh, UnsupportedVolumeGeometry, VolumeBackendUnavailable,
                       VolumeMesh, VolumeMeshingCancelled, generate_structured_box_mesh,
                       generate_volume_mesh, validate_volume_mesh)
from anymesher import structured_box

DELTA = 1e-7
BASE = [[0, 0, 0], [1, 0, 0], [1, .5, 0], [0, .5, 0]]
REVERSED = [BASE[0], BASE[3], BASE[2], BASE[1]]


def rotation(axis=(1., 2., 3.), angle=.7):
    k = np.asarray(axis, float)
    k /= np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def snapshot_of(profile, vector, offset=0., transform=None, shift=(0., 0., 0.)):
    profile, vector = np.asarray(profile, float), np.asarray(vector, float)
    if transform is not None:
        profile, vector = profile @ transform.T + shift, vector @ transform.T
    model = GeometryModel()
    region = model.add_solid_prism(profile.tolist(), vector.tolist(), offset=offset)
    return model.solid_region_snapshot(region)


def namespace_box(height=1.):
    x = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, height], [1, 0, height], [1, 1, height], [0, 1, height]], dtype=float)
    model_id = '71b52353-5ac5-4743-95b4-fc32c2347baf'
    refs = tuple(EntityHandle(model_id, 'face', i + 1) for i in range(6))
    loops = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return SimpleNamespace(
        schema_version=1, model_id=model_id, revision=7, region_id='solid-1', units='m',
        geometry_class='convex_normal_prism', coordinates=x, face_loops=loops, face_refs=refs,
        face_groups={'start': (refs[0],), 'end': (refs[1],), 'loaded': (refs[1], refs[2]),
                     'side_0': (refs[2],)},
        volume=height, bounds=np.array([x.min(axis=0), x.max(axis=0)]),
        source_signature='a' * 64, provenance={'fixture': 'analytic unit box'})


def generate(snapshot, target=.3, layers=3, **kwargs):
    return generate_structured_box_mesh(snapshot, target_edge_length=target, layers=layers,
                                        boundary_tolerance=kwargs.pop('delta', DELTA), **kwargs)


def independent_consumer_checks(mesh, snapshot):
    """Volume, coverage, outward ownership and corner checks from raw arrays only."""
    x, t = np.asarray(mesh.coordinates), np.asarray(mesh.cells)
    p = x[t]
    det = np.linalg.det(p[:, 1:] - p[:, :1])
    assert np.all(det > 0)
    volume = float(det.sum() / 6)
    assert volume == pytest.approx(snapshot.volume, rel=1e-9)
    rows = {tuple(row) for row in x.tolist()}
    assert all(tuple(c) in rows for c in np.asarray(snapshot.coordinates).tolist())
    centroid = np.asarray(snapshot.coordinates).mean(axis=0)
    ids = list(map(int, mesh.boundary_ids))
    for ref, loop in zip(snapshot.face_refs, snapshot.face_loops):
        polygon = np.asarray(snapshot.coordinates)[list(loop)]
        normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
        normal /= np.linalg.norm(normal)
        assert normal @ (polygon[0] - centroid) > 0, 'snapshot loop must be outward'
        area = float(np.linalg.norm(sum(np.cross(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))) / 2)
        owned = [i for i, key in enumerate(ids) if mesh.face_owners[key] == ref]
        tri = x[mesh.boundary_facets[owned]]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert np.all(cross @ normal > 0)
        assert float(np.linalg.norm(cross, axis=1).sum() / 2) == pytest.approx(area, rel=1e-10)
        assert np.max(np.abs((tri.reshape(-1, 3) - polygon[0]) @ normal)) <= 1e-12 * max(1., np.abs(x).max())
    assert sum(float(np.linalg.norm(np.cross(f[1] - f[0], f[2] - f[0]))) / 2
               for f in x[mesh.boundary_facets]) == pytest.approx(
        sum(float(np.linalg.norm(sum(np.cross(a, b) for a, b in zip(q, np.roll(q, -1, axis=0)))) / 2)
            for q in (np.asarray(snapshot.coordinates)[list(l)] for l in snapshot.face_loops)), rel=1e-10)


CASES = {
    'plain': dict(profile=BASE, vector=[0, 0, .25], divisions=(4, 2, 3)),
    'offset': dict(profile=BASE, vector=[0, 0, .25], offset=.125, divisions=(4, 2, 3)),
    'negative_extrusion': dict(profile=BASE, vector=[0, 0, -.25], divisions=(4, 2, 3)),
    'reversed_winding': dict(profile=REVERSED, vector=[0, 0, .25], divisions=(2, 4, 3)),
    'reversed_negative_offset': dict(profile=REVERSED, vector=[0, 0, -.25], offset=-.05, divisions=(2, 4, 3)),
    'rotated': dict(profile=BASE, vector=[0, 0, .25], offset=.1, transform=rotation(),
                    shift=(3., -2., 5.), divisions=(4, 2, 3)),
    'rotated_reversed': dict(profile=REVERSED, vector=[0, 0, -.25], transform=rotation((0, 1, 1), 2.1),
                             shift=(-4., 1., .5), divisions=(2, 4, 3)),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_orientation_offset_and_rotation_cases_are_admitted(name):
    case = dict(CASES[name])
    divisions = case.pop('divisions')
    snapshot = snapshot_of(**case)
    mesh = generate(snapshot)
    nx, ny, nz = divisions
    assert tuple(mesh.provenance['structured_box']['divisions']) == divisions
    assert len(mesh.coordinates) == (nx + 1) * (ny + 1) * (nz + 1)
    assert len(mesh.cells) == 6 * nx * ny * nz
    assert len(mesh.boundary_facets) == 4 * (nx * ny + nx * nz + ny * nz)
    assert mesh.provenance['orientation_swaps'] == 0
    assert mesh.provenance['source_signature'] == snapshot.source_signature
    assert mesh.provenance['contract'] == 'solid_volume_handoff_v1'
    assert snapshot.volume == pytest.approx(.125)
    independent_consumer_checks(mesh, snapshot)
    report = validate_volume_mesh(mesh, snapshot, boundary_tolerance=DELTA)
    assert report['volume'] == pytest.approx(.125, rel=1e-9)
    assert set(mesh.boundary_groups) == set(snapshot.face_groups)
    for name_, handles in snapshot.face_groups.items():
        ids = mesh.boundary_groups[name_]
        assert len(ids) and all(mesh.face_owners[int(i)] in handles for i in ids)
    assert np.array_equal(mesh.cell_groups[snapshot.region_id], mesh.cell_ids)


def test_in_plane_axes_are_sized_independently_and_layers_are_exact():
    snapshot = snapshot_of(BASE, [0, 0, .25])
    for target, layers, expected in [(1., 1, (1, 1, 1)), (.5, 2, (2, 1, 2)), (.3, 5, (4, 2, 5)),
                                     (.125, 1, (8, 4, 1))]:
        mesh = generate(snapshot, target, layers)
        assert tuple(mesh.provenance['structured_box']['divisions']) == expected
        edge = mesh.provenance['structured_box']['edge_lengths']
        assert edge[0] <= target * (1 + 1e-9) and edge[1] <= target * (1 + 1e-9)
        assert edge[2] == pytest.approx(.25 / layers)
    assert generate(snapshot, .5, np.int64(2)).provenance['structured_box']['layers'] == 2


def test_overlapping_groups_and_snapshot_identity_on_fixture_snapshot():
    snapshot = namespace_box()
    mesh = generate(snapshot, .5, 2)
    groups = {k: set(map(int, v)) for k, v in mesh.boundary_groups.items()}
    assert len(groups['start']) == len(groups['end']) == len(groups['side_0']) == 8
    assert groups['end'] < groups['loaded'] and groups['side_0'] < groups['loaded']
    assert len(groups['loaded']) == 16 and not groups['end'] & groups['side_0']
    assert (mesh.provenance['source_model_id'], mesh.provenance['source_revision'],
            mesh.provenance['source_region_id'], mesh.provenance['source_signature']) == (
        snapshot.model_id, 7, 'solid-1', 'a' * 64)
    assert mesh.provenance['source_provenance']['fixture'] == 'analytic unit box'
    independent_consumer_checks(mesh, snapshot)
    for key, value in [('revision', 8), ('source_signature', 'b' * 64), ('region_id', 'other')]:
        stale = namespace_box()
        setattr(stale, key, value)
        with pytest.raises(InvalidVolumeMesh, match='provenance'):
            validate_volume_mesh(mesh, stale, boundary_tolerance=DELTA)


def test_mesh_is_independent_immutable_and_persistable(tmp_path):
    snapshot = snapshot_of(BASE, [0, 0, .25], .125, rotation(), (3., -2., 5.))
    mesh = generate(snapshot, .5, 2)
    reference = mesh.coordinates.copy()
    for array in (mesh.coordinates, mesh.cells, mesh.boundary_facets, mesh.boundary_ids):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    path = tmp_path / 'mesh.npz'
    owners = sorted(mesh.face_owners.items())
    np.savez(path, coordinates=mesh.coordinates, cells=mesh.cells, node_ids=mesh.node_ids,
             cell_ids=mesh.cell_ids, facets=mesh.boundary_facets, boundary_ids=mesh.boundary_ids,
             owner_keys=[k for k, _ in owners], owner_faces=[h.id for _, h in owners],
             **{f'group_{k}': v for k, v in mesh.boundary_groups.items()})
    with np.load(path) as data:
        restored = VolumeMesh(
            data['coordinates'], data['cells'], data['node_ids'], data['cell_ids'],
            data['facets'], data['boundary_ids'],
            cell_groups={snapshot.region_id: data['cell_ids']},
            boundary_groups={k[6:]: data[k] for k in data.files if k.startswith('group_')},
            face_owners={int(k): EntityHandle(snapshot.model_id, 'face', int(i))
                         for k, i in zip(data['owner_keys'], data['owner_faces'])},
            provenance=mesh.provenance)
    assert validate_volume_mesh(restored, snapshot, boundary_tolerance=DELTA)['cells'] == len(mesh.cells)
    assert np.array_equal(mesh.coordinates, reference)
    # Repeated generation is deterministic.
    again = generate(snapshot, .5, 2)
    for name in ('coordinates', 'cells', 'boundary_facets', 'boundary_ids'):
        assert np.array_equal(getattr(mesh, name), getattr(again, name))


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, None, 'x', [], 1 + 2j])
def test_invalid_target_edge_length(value):
    with pytest.raises(InvalidVolumeMesh):
        generate(snapshot_of(BASE, [0, 0, .25]), value)


@pytest.mark.parametrize('value', [0, -1, True, 1.5, 2., None, '2', [2], float('nan')])
def test_invalid_layers(value):
    with pytest.raises(InvalidVolumeMesh, match='layers'):
        generate(snapshot_of(BASE, [0, 0, .25]), .5, value)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, None, 'x', 1e-300, 10.])
def test_invalid_boundary_tolerance(value):
    with pytest.raises(InvalidVolumeMesh):
        generate(snapshot_of(BASE, [0, 0, .25]), .5, 1, delta=value)


def test_unsupported_topology_has_owner_error():
    triangle = snapshot_of([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 0, 1])
    pentagon = snapshot_of([[0, 0, 0], [2, 0, 0], [3, 1, 0], [1, 3, 0], [-1, 1, 0]], [0, 0, 1])
    trapezoid = snapshot_of([[0, 0, 0], [2, 0, 0], [1.5, 1, 0], [.5, 1, 0]], [0, 0, 1])
    parallelogram = snapshot_of([[0, 0, 0], [1, 0, 0], [1.5, 1, 0], [.5, 1, 0]], [0, 0, 1])
    for snapshot in (triangle, pentagon, trapezoid, parallelogram):
        with pytest.raises(UnsupportedVolumeGeometry):
            generate(snapshot, .5, 1)
    concave = namespace_box()
    concave.geometry_class = 'concave'
    with pytest.raises(UnsupportedVolumeGeometry):
        generate(concave, .5, 1)
    swapped = namespace_box()
    swapped.face_loops = (swapped.face_loops[1], swapped.face_loops[0], *swapped.face_loops[2:])
    with pytest.raises(UnsupportedVolumeGeometry, match='layout'):
        generate(swapped, .5, 1)
    malformed = namespace_box()
    malformed.revision = True
    with pytest.raises(InvalidVolumeMesh):
        generate(malformed, .5, 1)


def test_resource_guard_refuses_impossible_lattices_before_allocation(monkeypatch):
    snapshot = snapshot_of(BASE, [0, 0, .25])
    def forbidden(*args, **kwargs):
        raise AssertionError('admission must not be reached')
    monkeypatch.setattr(structured_box, '_admit_provider_output', forbidden)
    real_empty = np.empty
    def guarded_empty(shape, *args, **kwargs):
        assert np.prod(shape) < 10_000, 'impossible lattice allocation attempted'
        return real_empty(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'empty', guarded_empty)
    for target, layers in [(1e-12, 1), (1e-6, 1), (.5, 10 ** 12), (.5, np.int64(10 ** 15)), (1e-300, 2)]:
        with pytest.raises(InvalidVolumeMesh, match='resource limit'):
            generate(snapshot, target, layers)
    monkeypatch.setattr(structured_box, 'MAX_STRUCTURED_BOX_CELLS', 100)
    with pytest.raises(InvalidVolumeMesh, match='cell resource limit'):
        generate(snapshot, .125, 3)


def test_cancellation_during_generation_and_admission_is_atomic():
    snapshot = snapshot_of(BASE, [0, 0, .25])
    with pytest.raises(VolumeMeshingCancelled):
        generate(snapshot, .5, 1, cancel=lambda: True)
    calls = [0]
    def count():
        calls[0] += 1
        return False
    baseline = generate(snapshot, .5, 2, cancel=count)
    total = calls[0]
    assert total > 10, 'generation and admission must both poll cancellation'
    for stop in range(1, total + 1):
        seen = [0]
        def cancel(stop=stop):
            seen[0] += 1
            return seen[0] >= stop
        with pytest.raises(VolumeMeshingCancelled):
            generate(snapshot, .5, 2, cancel=cancel)

    class Event:
        def __init__(self):
            self.n = 0
        def is_set(self):
            self.n += 1
            return self.n > 3
    with pytest.raises(VolumeMeshingCancelled):
        generate(snapshot, .5, 2, cancel=Event())
    assert len(baseline.cells) == 6 * 2 * 1 * 2


def test_native_gate_and_default_volume_api_are_unchanged(monkeypatch):
    assert anymesher.generate_structured_box_mesh is structured_box.generate_structured_box_mesh
    assert 'generate_structured_box_mesh' in anymesher.__all__
    parameters = inspect.signature(generate_volume_mesh).parameters
    assert parameters['backend'].default == 'pytetwild'
    assert list(inspect.signature(generate_structured_box_mesh).parameters) == [
        'snapshot', 'target_edge_length', 'layers', 'boundary_tolerance', 'cancel']
    kinds = inspect.signature(generate_structured_box_mesh).parameters
    assert all(kinds[k].kind is inspect.Parameter.KEYWORD_ONLY for k in list(kinds)[1:])
    calls = []
    monkeypatch.setattr('anymesher._volume_process.run_provider', lambda *a: calls.append(a))
    snapshot = snapshot_of(BASE, [0, 0, .25])
    generate(snapshot, .5, 1)
    with pytest.raises(VolumeBackendUnavailable):
        generate_volume_mesh(snapshot, target_edge_length=.5, boundary_tolerance=DELTA, backend='other')
    assert not calls
    with pytest.raises(TypeError):
        generate_structured_box_mesh(snapshot, .5, 1, DELTA)


def test_optional_anysolid_consumer_resultants():
    anysolid = pytest.importorskip('anysolid')
    snapshot = snapshot_of(BASE, [0, 0, .25], .125, rotation(), (3., -2., 5.))
    mesh = generate(snapshot, .5, 2)
    solid = anysolid.SolidMesh.snapshot(mesh)
    prepared = anysolid.prepare(solid, {'elastic_modulus': 1000., 'poisson_ratio': .3}, order=1)
    body = np.array([2., -3., 5.])
    resultant = np.bincount(prepared.coefficient_components, weights=prepared.body_load(body), minlength=3)
    np.testing.assert_allclose(resultant, snapshot.volume * body, rtol=1e-10, atol=1e-12)
