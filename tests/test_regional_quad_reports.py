"""Quadratic publication transports midside labels independently of source geometry IDs."""
from types import SimpleNamespace

import pytest
from anymesher._regional_quad_reports import _high_order
from anymesher.errors import MeshError
from anymesher.quad.high_order import HighOrderBoundaryMidside, HighOrderGeometryReport, HighOrderMeshCertificate
from anymesher.regional_update import _Ids


def _certificate():
    mid = HighOrderBoundaryMidside((5,0.,1.), 5, (0.,1.), .5, 5, 0., 'straight')
    face = HighOrderGeometryReport('test', 2, 1, 'cylindrical', 'cylindrical',
        (0.,0.,0.), 'curve', 'surface', 1, 0, 1, 1, (mid,), ('straight',), 3, 0.)
    return HighOrderMeshCertificate('CERTIFIED_POSITIVE', 'test', 2, (face,), 1, 0, 4, 1, 0.).to_dict()


@pytest.mark.parametrize('invalid', (False, True))
def test_boundary_midside_translates_while_source_edge_and_interval_survive(invalid):
    nodes = set(range(1,9))
    mesh = SimpleNamespace(is_quadratic=True, quads={1:tuple(range(1,9))}, tris={},
                           elements_of_face={1:[1]}, nodes_of_edge={5:[1,5,2]})
    ids = _Ids(nodes, {1}, set(), {n:100+n for n in nodes}, {1:20}, quads={1})
    source = _certificate()
    if invalid:
        source['reports'][0]['boundary_midsides'][0]['midpoint_node'] = 1
        with pytest.raises(MeshError, match='registered source edge'):
            _high_order(source, ids, mesh)
    else:
        result = _high_order(source, ids, mesh)
        station = result['boundary_midsides'][0]
        assert station['midpoint_node'] == 105
        assert station['source_edge'] == 5
        assert station['canonical_edge'] == [5,0.,1.]
        assert result['reports'][0]['boundary_midsides'] == [station]
        assert source['boundary_midsides'][0]['midpoint_node'] == 5


def test_q5_transports_only_published_node_ids():
    from anymesher._regional_quad_reports import _map_q5
    from anymesher.quad.optimize import QuadOptimizerReport
    base = QuadOptimizerReport('APPLIED', 1, 1, 1, 1, 1., .5, (8,), .125, ('APPLIED',), 2).to_dict()
    local = dict(base, resident_moved_node_ids=[3], transition_quality={
        'evaluation_budget': 10, 'evaluations': 4, 'sweeps': 1,
        'final_excess': .5, 'initial_excess': 1., 'status': 'APPLIED',
        'moved_node_ids': [3], 'global_moved_node_ids': [8]})
    report = dict(base, faces={7: local})
    _map_q5(report, _Ids({8}, set(), set(), {8: 30}, {}))
    assert report['moved_node_ids'] == local['moved_node_ids'] == [30]
    assert local['transition_quality']['global_moved_node_ids'] == [30]
    assert local['resident_moved_node_ids'] == local['transition_quality']['moved_node_ids'] == [3]


def _merge_shaped_mesh():
    """A quad-first plate carrying the diagnostics shape of hybrid's quad-first/legacy merge.

    ``hybrid._merge_quad_first_and_legacy`` publishes ``quad_first`` (the quad route's own
    diagnostics), ``quad_first_face_ids``/``legacy_face_ids`` (geometry ids) and one top-level
    ``"<face>"`` backend record per legacy face.
    """
    from dataclasses import replace

    from anygeometry import GeometryModel
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.quad.options import QuadMeshingOptions

    geometry = GeometryModel()
    face = geometry.add_plate(geometry.add_points([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]))
    legacy_face = geometry.add_plate(geometry.add_points([(2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)]))
    mesh = generate_hybrid_mesh_result(geometry, target_size=.25, face_ids=(face,),
                                       quad_options=QuadMeshingOptions(), layout_policy="adaptive").mesh
    first = min(mesh.nodes)
    quad = mesh.hybrid_diagnostics
    quad["q5"]["moved_node_ids"] = [first]
    for row in quad["q5"]["faces"].values():
        row["moved_node_ids"] = [first]
    # The per-face record is the one the legacy producer really emits.
    legacy = generate_hybrid_mesh_result(geometry, target_size=.25, face_ids=(legacy_face,))
    backend = dict(legacy.triangulation_backend_by_face[legacy_face])
    merged = {str(legacy_face): backend, "quad_first_face_ids": [face],
              "legacy_face_ids": [legacy_face], "quad_first": quad}
    return replace(mesh, hybrid_diagnostics=merged), face, legacy_face, first


def _legacy_record(diagnostics):
    return next(v for k, v in diagnostics.items() if isinstance(k, str) and k.isdecimal())


def _numbering(mesh):
    return ({n: 101 + 2 * n for n in mesh.nodes}, {e: 501 + 3 * e for e in mesh.shells})


def test_merged_quad_first_legacy_diagnostics_renumber_with_true_geometry_ids():
    import copy

    from anymesher.regional_update import remap_mesh_ids

    mesh, face, legacy_face, first = _merge_shaped_mesh()
    before = copy.deepcopy(mesh.hybrid_diagnostics)
    nodes, elements = _numbering(mesh)
    numbered = remap_mesh_ids(mesh, nodes, elements)
    out = numbered.hybrid_diagnostics
    assert out["quad_first_face_ids"] == [face] and out["legacy_face_ids"] == [legacy_face]
    assert out[str(legacy_face)] == before[str(legacy_face)]
    nested = out["quad_first"]
    assert nested["q5"]["moved_node_ids"] == [nodes[first]]
    assert all(row["moved_node_ids"] == [nodes[first]] for row in nested["q5"]["faces"].values())
    assert nested["front"] == before["quad_first"]["front"]
    assert set(nested["front"]["faces"]) == {face}
    assert mesh.hybrid_diagnostics == before  # the source is not mutated


@pytest.mark.parametrize("damage, message", (
    (lambda d: d["quad_first"].update(unregistered=[1]), "no explicit schema"),
    (lambda d: d["quad_first"].update(quad_first={}), "no explicit schema"),
    (lambda d: d.update(unregistered=[1]), "no explicit schema"),
    (lambda d: d.update({"999999": _legacy_record(d)}), "legacy face diagnostics"),
    (lambda d: d.pop("legacy_face_ids"), "incomplete quad-first/legacy"),
    (lambda d: d.pop("quad_first"), "incomplete quad-first/legacy"),
    (lambda d: d.update(legacy_face_ids=d["quad_first_face_ids"] + d["legacy_face_ids"]), "overlap"),
    (lambda d: _legacy_record(d).update(node_ids=[1]), "mapped-face record"),
    (lambda d: d["quad_first"]["q5"].update(moved_node_ids=[10 ** 6]), "not a node"),
    (lambda d: d["quad_first"]["q5"].update(moved_node_ids=["1"]), "integer"),
))
def test_merged_diagnostics_fail_closed_on_unknown_or_ambiguous_evidence(damage, message):
    from anymesher.regional_update import remap_mesh_ids

    mesh, *_ = _merge_shaped_mesh()
    damage(mesh.hybrid_diagnostics)
    nodes, elements = _numbering(mesh)
    with pytest.raises(MeshError, match=message):
        remap_mesh_ids(mesh, nodes, elements)


def test_nested_quad_first_ids_are_limited_to_the_quad_shell_submesh():
    import copy

    import numpy as np
    from anymesher.regional_update import remap_mesh_ids

    mesh, *_ = _merge_shaped_mesh()
    outside = max(mesh.nodes) + 1  # a real node of the mesh (e.g. a beam station) but not a shell node
    mesh.nodes[outside] = np.zeros(3)
    nodes, elements = _numbering(mesh)
    nested = mesh.hybrid_diagnostics["quad_first"]
    good = copy.deepcopy(mesh.hybrid_diagnostics)
    assert remap_mesh_ids(mesh, nodes, elements).hybrid_diagnostics["quad_first"]["q5"]["moved_node_ids"]
    nested["q5"]["moved_node_ids"] = [outside]
    with pytest.raises(MeshError, match="not a node"):
        remap_mesh_ids(mesh, nodes, elements)
    nested["q5"]["moved_node_ids"] = good["quad_first"]["q5"]["moved_node_ids"]
    for row in nested["q5"]["faces"].values():
        row["moved_node_ids"] = [outside]
    with pytest.raises(MeshError, match="not a node"):
        remap_mesh_ids(mesh, nodes, elements)


def test_flat_and_wrapped_quad_first_evidence_cannot_contradict():
    import copy

    from anymesher.regional_update import remap_mesh_ids

    mesh, face, legacy_face, first = _merge_shaped_mesh()
    quad = mesh.hybrid_diagnostics["quad_first"]
    mesh.hybrid_diagnostics.update(copy.deepcopy(quad))  # the through-beam route emits both
    nodes, elements = _numbering(mesh)
    out = remap_mesh_ids(mesh, nodes, elements).hybrid_diagnostics
    assert out["q5"] == out["quad_first"]["q5"] and out["high_order_geometry"] == out["quad_first"]["high_order_geometry"]
    other = max(mesh.nodes)
    assert other != first
    mesh.hybrid_diagnostics["q5"]["moved_node_ids"] = [other]
    with pytest.raises(MeshError, match="flat and wrapped"):
        remap_mesh_ids(mesh, nodes, elements)
    mesh.hybrid_diagnostics["q5"]["moved_node_ids"] = [first]
    mesh.hybrid_diagnostics["unregistered"] = [1]
    with pytest.raises(MeshError, match="no explicit schema"):
        remap_mesh_ids(mesh, nodes, elements)


def _two_station_certificate(nodes):
    mids = (HighOrderBoundaryMidside((5, 0., .5), 5, (0., .5), .25, nodes[0], 0., 'straight'),
            HighOrderBoundaryMidside((5, .5, 1.), 5, (.5, 1.), .75, nodes[1], 0., 'straight'))
    face = HighOrderGeometryReport('test', 2, 1, 'cylindrical', 'cylindrical',
        (0., 0., 0.), 'curve', 'surface', 1, 0, 1, 1, mids, ('straight',), 2, 0.)
    return HighOrderMeshCertificate('CERTIFIED_POSITIVE', 'test', 2, (face,), 1, 0, 4, 2, 0.).to_dict()


def _station_mesh():
    return SimpleNamespace(is_quadratic=True, quads={1: tuple(range(1, 9))}, tris={},
                           elements_of_face={1: [1]}, nodes_of_edge={5: [1, 6, 2, 7, 3]})


def test_boundary_midside_must_be_the_actual_edge_station_before_and_after_sparse_ids():
    mesh = _station_mesh()
    sparse = {n: 1000 + 13 * n for n in range(1, 9)}
    ids = _Ids(set(sparse), {1}, set(), sparse, {1: 20}, quads={1})
    result = _high_order(_two_station_certificate((6, 7)), ids, mesh)
    assert [row['midpoint_node'] for row in result['boundary_midsides']] == [sparse[6], sparse[7]]
    # Both nodes are registered midsides of the edge, but swapped between stations.
    with pytest.raises(MeshError, match="published chain midside"):
        _high_order(_two_station_certificate((7, 6)), ids, mesh)
    # Face sources that traverse the edge in opposite orientation list stations in
    # opposite order; the station interval, not the row order, identifies the node.
    reversed_rows = _two_station_certificate((6, 7))
    reversed_rows['reports'][0]['boundary_midsides'].reverse()
    flipped = _high_order(reversed_rows, ids, mesh)
    assert flipped['boundary_midsides'] == result['boundary_midsides']
    assert [r['midpoint_node'] for r in flipped['reports'][0]['boundary_midsides']] == [sparse[7], sparse[6]]
    # A chain that does not carry exactly one midside per certified station is refused.
    longer = _station_mesh()
    longer.nodes_of_edge[5] = [1, 6, 2, 7, 3, 8, 4]
    with pytest.raises(MeshError, match="certifies 2 stations"):
        _high_order(_two_station_certificate((6, 7)), ids, longer)


def test_regional_splice_still_refuses_merged_quad_first_evidence():
    from anymesher.regional_update import _validate_metadata

    mesh, *_ = _merge_shaped_mesh()
    evidence = _validate_metadata(mesh, "previous mesh", set(mesh.nodes), set(mesh.shells))
    assert "hybrid_diagnostics.quad_first" in evidence  # assemble_regional_replacement refuses it unless discarded


def test_producer_translation_covers_zero_and_renumbered_nodes():
    from anymesher.regional_update import translate_hybrid_diagnostics

    mesh, face, legacy_face, first = _merge_shaped_mesh()
    from dataclasses import replace
    quad = mesh.hybrid_diagnostics["quad_first"]
    shell = replace(mesh, hybrid_diagnostics=quad)
    ordered = sorted(shell.nodes)
    reverse = {n: m for n, m in zip(ordered, reversed(ordered))}
    identity = {e: e for e in shell.shells}
    translated = translate_hybrid_diagnostics(shell, reverse, identity)
    assert translated["q5"]["moved_node_ids"] == [reverse[first]]
    assert quad["q5"]["moved_node_ids"] == [first]
    with pytest.raises(MeshError, match="injective"):
        translate_hybrid_diagnostics(shell, {n: ordered[0] for n in ordered}, identity)


def test_edge_aligned_seed_counts_survive_sparse_numbering():
    from anygeometry import GeometryModel
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.quad.options import QuadMeshingOptions
    from anymesher.regional_update import remap_mesh_ids

    geometry = GeometryModel()
    face = geometry.add_plate(geometry.add_points([(0,0,0), (1,0,0), (1,1,0), (0,1,0)]))
    mesh = generate_hybrid_mesh_result(geometry, target_size=.25, face_ids=(face,),
                                       quad_options=QuadMeshingOptions(), layout_policy="adaptive").mesh
    seed = mesh.hybrid_diagnostics["front"]["faces"][face]["edge_aligned_seed"]
    nodes = {n: 101 + 2*n for n in mesh.nodes}
    elements = {e: 501 + 3*e for e in mesh.shells}
    numbered = remap_mesh_ids(mesh, nodes, elements)
    assert numbered.hybrid_diagnostics["front"]["faces"][face]["edge_aligned_seed"] == seed
    assert set(numbered.shells) == set(elements.values())
    seed["unregistered_node_ids"] = [1]
    with pytest.raises(MeshError, match="unknown or missing fields"):
        remap_mesh_ids(mesh, nodes, elements)
    del seed["unregistered_node_ids"]
    seed["tensor_cells"] = -1
    with pytest.raises(MeshError, match="non-negative"):
        remap_mesh_ids(mesh, nodes, elements)
