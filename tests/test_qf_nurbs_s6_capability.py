"""S6: the advertised quad-first scope names the NURBS tensor route instead of a blanket 'curved' exclusion.

The workers are not built in this copy (``advertise_quad_capabilities`` raises ``QuadCapabilityMissing``), so the
wording is pinned at its two definition sites: the keyword default and the report built by the public router.
"""
import inspect

import anymesher.quad.public_integration as module

WORDING = ("curved: NURBS tensor-eligible faces only", "Q9+")


def test_advertise_default_wording():
    assert inspect.signature(module.advertise_quad_capabilities).parameters['unsupported_scope'].default == WORDING


def test_public_router_report_wording():
    source = inspect.getsource(module)
    assert source.count('unsupported_scope=("curved: NURBS tensor-eligible faces only", "Q9+")') == 1
    assert 'unsupported_scope=("curved", "Q9+")' not in source


def test_report_carries_the_wording():
    report = module.QuadCapabilityReport(quad_first_api="public/1", compiled_native_support=False,
        front_path="advancing_front", q4_count_worker="x", q5_tinyad_worker="x", mixed_q4_s3=False,
        unsupported_scope=WORDING)
    assert report.to_dict()["unsupported_scope"] == WORDING
