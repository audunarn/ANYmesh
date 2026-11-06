"""Quality targets are distinct from topology validity and explicit policies."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anymesher import NativeMeshingOptions
from anymesher import _cylindrical_public as public
from anymesher.core import MeshCore
from anymesher.errors import MeshError
from anymesher.hybrid import validate_mesh_quality
from anymesher.mesh import Mesh
from anymesher.quality_v2 import assert_valid_mesh, evaluate_quality
from anymesher.structured import StructuredMeshingOptions
from anymesher.surface_mesh import SurfaceMeshOptions, _quality_threshold_report, mesh_planar_surface


def settings(policy=None):
    size = SimpleNamespace(target_size=.4, zones=(), _sources=())
    return public._quadratic_settings(size, NativeMeshingOptions(), policy, False, 'python')


def test_default_quadratic_settings_keep_30_as_unenforced_target():
    value = settings()
    assert value.min_angle == 30.
    assert value.enforce_quality is False
    assert value.max_element_growth == 1.5


def test_explicit_quadratic_settings_preserve_all_requested_policy_fields():
    policy = StructuredMeshingOptions(quality_policy=dict(minimum_scaled_jacobian=.3,
        maximum_aspect_ratio=3.7, minimum_angle=30., maximum_angle=142., maximum_warpage=.06),
        max_element_growth=1.23)
    value = settings(policy)
    assert value.enforce_quality is True
    assert (value.min_scaled_jacobian, value.max_aspect_ratio, value.min_angle,
            value.max_angle, value.max_warpage, value.max_element_growth) == (.3, 3.7, 30., 142., .06, 1.23)


def poor_angle_core():
    # Positive, conforming triangle: minimum26.565 degrees, all other default
    # shape targets met. No tolerance/search or 15-degree replacement.
    return MeshCore(np.array(((0., 0., 0.), (1., 0., 0.), (.5, .25, 0.))),
                    np.array(((0, 1, 2),)))


def test_real_valid_target_miss_is_reported_but_only_explicit_enforcement_rejects():
    core = poor_angle_core()
    assert_valid_mesh(core)
    value = SurfaceMeshOptions(backend='python', recombine=False)
    report = _quality_threshold_report(evaluate_quality(core), value)
    assert report['accepted'] is False
    assert report['violation_counts'] == dict(scaled_jacobian=0, aspect_ratio=0,
        minimum_angle=1, maximum_angle=0, warpage=0)
    produced = mesh_planar_surface(core.node_coordinates[:, :2], options=value, _boundary_is_seeded=True)
    assert_valid_mesh(produced)
    assert _quality_threshold_report(evaluate_quality(produced), value)['accepted'] is False
    with pytest.raises(MeshError, match='mesh quality policy rejected'):
        mesh_planar_surface(core.node_coordinates[:, :2],
            options=replace(value, enforce_quality=True), _boundary_is_seeded=True)


def test_invalid_topology_cannot_become_a_quality_target_warning():
    core = MeshCore(np.array(((0., 0., 0.), (1., 0., 0.), (.5, 0., 0.))),
                    np.array(((0, 1, 2),)))
    with pytest.raises(MeshError):
        assert_valid_mesh(core)


def mesh(points, triangles):
    return Mesh(geometry_model_id='quality-contract-unit', geometry_revision=0,
        nodes={i + 1: np.array(p) for i, p in enumerate(points)},
        tris={i + 1: tuple(n + 1 for n in row) for i, row in enumerate(triangles)},
        elements_of_face={1: list(range(1, len(triangles) + 1))})


def test_explicit_public_angle_policy_still_refuses_real_target_miss():
    value = mesh(((0., 0., 0.), (1., 0., 0.), (.5, .25, 0.)), ((0, 1, 2),))
    assert validate_mesh_quality(None, value) is None
    with pytest.raises(MeshError, match='explicit mesh quality policy failed'):
        validate_mesh_quality(None, value,
            quality_options=StructuredMeshingOptions(quality_policy={'minimum_angle': 30.}))


def test_explicit_public_growth_policy_still_refuses_real_violation():
    # Every shape angle exceeds30; only the adjacent perimeter-growth target
    # exceeds1.5. Explicit policy remains independently authoritative.
    value = mesh(((0., 0., 0.), (1., 0., 0.), (.5, np.sqrt(3.) / 2, 0.), (.5, -1.8, 0.)),
                 ((0, 1, 2), (1, 0, 3)))
    with pytest.raises(MeshError, match='growth violations='):
        validate_mesh_quality(None, value,
            quality_options=StructuredMeshingOptions(quality_policy={'minimum_angle': 30.}, max_element_growth=1.5))
