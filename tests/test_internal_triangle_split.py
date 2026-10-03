"""Owner-declared incidence and conformity, independent of geometry sampling."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from anymesher._shared_triangle_split import propagate_triangle_split
from anymesher.errors import MeshError
from anymesher.mesh import Mesh


def fixture():
    # Face10 crosses the station edge; face20 terminates there in another plane.
    return Mesh(
        nodes={1: np.array((0., 0., 0.)), 2: np.array((2., 0., 0.)),
               3: np.array((1., 1., 0.)), 4: np.array((1., -1., 0.)),
               5: np.array((1., 0., 1.)), 6: np.array((1., 0., 0.)),
               7: np.array((.5, 0., 0.))},
        tris={10: (1, 2, 3), 11: (2, 1, 4), 20: (2, 1, 5)},
        elements_of_face={10: [10, 11], 20: [20]},
    )


def incidence(mesh, face, edge):
    return sum(set(edge).issubset(mesh.tris[element])
               for element in mesh.elements_of_face[face])


def area_vector(mesh, face):
    total = [Fraction(0)] * 3
    for element in mesh.elements_of_face[face]:
        a, b, c = ([Fraction(float(v)) for v in mesh.nodes[n]]
                   for n in mesh.tris[element])
        u, v = ([b[i] - a[i] for i in range(3)],
                [c[i] - a[i] for i in range(3)])
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2],
                 u[0]*v[1]-u[1]*v[0])
        total = [total[i] + cross[i] for i in range(3)]
    return tuple(total)


def snapshot(mesh, cache):
    return (dict(mesh.tris), deepcopy(mesh.elements_of_face), deepcopy(cache),
            {node: point.tobytes() for node, point in mesh.nodes.items()})


def test_internal_and_boundary_owners_keep_exact_area_and_station_sequences():
    mesh, cache = fixture(), {}
    original = snapshot(mesh, cache)
    areas = {face: area_vector(mesh, face) for face in (10, 20)}
    assert propagate_triangle_split(mesh, (20, 10), (2, 1), 6,
                                   cache=cache, interior_face_ids=(10,)) == 3
    assert propagate_triangle_split(mesh, (10, 20), (1, 6), 7,
                                   cache=cache, interior_face_ids=(10,)) == 3
    for face, count in ((10, 2), (20, 1)):
        assert area_vector(mesh, face) == areas[face]
        assert incidence(mesh, face, (1, 2)) == 0
        for edge in ((1, 7), (7, 6), (6, 2)):
            assert incidence(mesh, face, edge) == count
            assert len(cache[face][tuple(sorted(edge))]) == count
    assert {n: p.tobytes() for n, p in mesh.nodes.items()} == original[3]


def test_face_order_and_endpoint_orientation_are_deterministic():
    first, second = fixture(), fixture()
    a, b = {}, {}
    propagate_triangle_split(first, (10, 20), (1, 2), 6, cache=a,
                             interior_face_ids=(10,))
    propagate_triangle_split(second, (20, 10, 10), (2, 1), 6, cache=b,
                             interior_face_ids=(10,))
    assert snapshot(first, a) == snapshot(second, b)


@pytest.mark.parametrize('interior, message', (((), 'exactly one'),
    ((10, 20), 'exactly two'), ((30,), 'unselected')))
def test_incidence_never_grants_split_permission(interior, message):
    mesh, cache = fixture(), {}
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError, match=message):
        propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=interior)
    assert snapshot(mesh, cache) == before


@pytest.mark.parametrize('identity', (True, False, -10, 10., '10'))
def test_invalid_interior_identity_never_grants_permission(identity):
    mesh, cache = fixture(), {}
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError, match='positive face identities'):
        propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=(identity,))
    assert snapshot(mesh, cache) == before


@pytest.mark.parametrize('identities', ((10, 10.), (10., 10), (1, True),
                                      (True, 1), (10, np.float64(10)), ([10],)))
def test_duplicate_permission_never_hides_an_invalid_original_entry(identities):
    mesh, cache = fixture(), {}
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError, match='positive face identities'):
        propagate_triangle_split(mesh, (1, 10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=identities)
    assert snapshot(mesh, cache) == before


def test_declared_built_interior_owner_cannot_silently_disappear():
    mesh, cache = fixture(), {}
    mesh.elements_of_face[10] = []
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError, match='no active triangles'):
        propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=(10,))
    assert snapshot(mesh, cache) == before


def test_default_boundary_route_retains_unbuilt_owner_behavior():
    mesh, cache = fixture(), {}
    mesh.elements_of_face[10] = []
    assert propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache) == 1
    assert 10 not in cache


@pytest.mark.parametrize('cached', (False, True))
def test_invalid_later_owner_preserves_topology_and_cache(cached):
    mesh, cache = fixture(), {}
    if cached:
        propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=(10,))
        edge, station = (1, 6), 7
        mesh.nodes[5] = mesh.nodes[1].copy()  # Degenerate later-owner geometry.
    else:
        edge, station = (1, 2), 6
        mesh.tris[20] = (2, 3, 5)  # Does not use the registered interval.
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError):
        propagate_triangle_split(mesh, (10, 20), edge, station, cache=cache,
                                 interior_face_ids=(10,))
    assert snapshot(mesh, cache) == before


@pytest.mark.parametrize('phase', ('shared triangle split owner staging',
    'shared triangle split incidence staging', 'shared triangle split commit'))
def test_cancellation_preserves_exact_exception_and_all_owned_state(phase):
    mesh, cache = fixture(), {}
    before = snapshot(mesh, cache)
    sentinel = MeshError('cancelled')
    def cancel(current):
        if current == phase:
            raise sentinel
    with pytest.raises(MeshError) as caught:
        propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=(10,), cancellation_check=cancel)
    assert caught.value is sentinel
    assert snapshot(mesh, cache) == before


def test_internal_owner_must_have_opposite_edge_directions():
    mesh, cache = fixture(), {}
    mesh.tris[11] = (1, 2, 4)
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError, match='inconsistent face orientation'):
        propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=(10,))
    assert snapshot(mesh, cache) == before


def test_quadratic_neighbour_is_not_silently_subdivided():
    mesh, cache = fixture(), {}
    mesh.tris[20] = (2, 1, 5, 6, 7, 3)
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError, match='linear T3'):
        propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                                 interior_face_ids=(10,))
    assert snapshot(mesh, cache) == before


@pytest.mark.parametrize('mutation', ('quadratic', 'missing_face', 'wrong_edge'))
def test_affected_stale_cache_parent_refuses_without_losing_topology(mutation):
    mesh, cache = fixture(), {}
    propagate_triangle_split(mesh, (10, 20), (1, 2), 6, cache=cache,
                             interior_face_ids=(10,))
    parent = min(cache[10][(1, 6)])
    if mutation == 'quadratic':
        mesh.tris[parent] += (4, 5, 7)
    elif mutation == 'missing_face':
        mesh.elements_of_face[10].remove(parent)
    else:
        mesh.tris[parent] = (2, 3, 4)
    before = snapshot(mesh, cache)
    with pytest.raises(MeshError, match='linear T3|parent interval'):
        propagate_triangle_split(mesh, (10, 20), (1, 6), 7, cache=cache,
                                 interior_face_ids=(10,))
    assert snapshot(mesh, cache) == before


@pytest.mark.parametrize('warm', (False, True))
def test_cancellation_after_first_owner_staging_does_not_publish_cache(warm):
    mesh, cache = fixture(), {}
    edge, station = (1, 2), 6
    if warm:
        propagate_triangle_split(mesh, (10, 20), edge, station, cache=cache,
                                 interior_face_ids=(10,))
        edge, station = (1, 6), 7
    before = snapshot(mesh, cache)
    sentinel = RuntimeError('second owner cancelled')
    calls = 0
    def cancel(phase):
        nonlocal calls
        if phase == 'shared triangle split owner staging':
            calls += 1
            if calls == 2:
                raise sentinel
    with pytest.raises(RuntimeError) as caught:
        propagate_triangle_split(mesh, (10, 20), edge, station, cache=cache,
                                 interior_face_ids=(10,), cancellation_check=cancel)
    assert caught.value is sentinel
    assert calls == 2
    assert snapshot(mesh, cache) == before
