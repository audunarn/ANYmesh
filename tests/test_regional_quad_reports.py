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
