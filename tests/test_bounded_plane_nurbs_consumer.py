"""Bounded crossing through public geometry, meshing and live source proof.

Acceptance definitions, AUTHORED NOT RUN. No geometry test imports or synthetic
receipts. The original two-parent publication boundary is separately documented
in run21-bounded-consumer-HANDOFF.md; a prepared-source S3 is not that publication.
"""
from collections import defaultdict
from copy import deepcopy

import pytest
import anygeometry as owner
from anygeometry.entities import EntityRef
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anymesher import (validate_material_source_transfer_for_mesh,
    qualify_model_closure_source_publication, publish_model_closure_mesh_to_source)
from anymesher.errors import MeshError
from anymesher.quality import verify_mesh_quality
from anymesher.recovery import (
    generate_automatic_mesh_result, select_curvilinear_recovery_policy,
)


def _step(phase):
    """Optional evidence-only stream; no effect on proof or meshing options."""
    import json
    import os
    import time
    from pathlib import Path
    destination = os.environ.get('NURBS_STEP_EVIDENCE')
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase':str(phase), 'monotonic':time.perf_counter()})+'\n')


def crossing_source():
    model = owner.GeometryModel(tolerance=TolerancePolicy(length=1e-9, surface_residual=1e-8))
    heights = (1., -1., 1.)
    support = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.),
        (0., 0., 0., 1., 1., 1.), tuple(tuple(
            (i-1., j-1., heights[i]+heights[j]) for j in range(3)) for i in range(3)))
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    vertices = [model.add_point(*support.evaluate(*p)) for p in uv]
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i+1)%4])))) for i, p in enumerate(uv)]
    bowl = model.add_face(edges, surface=support)
    vertices = [model.add_point(x, y, .5) for x, y in
                ((.35, -1.), (1.5, -1.), (1.5, 1.), (.35, 1.))]
    plate = model.add_plate(vertices)
    for face, name, thickness in ((bowl, 'bowl-parent', .01), (plate, 'plate-parent', .02)):
        model.update_face_metadata(face, engineering_parent=name, thickness=thickness)
        model.add_sheet((face,), name=name)
    _step('source built')
    return model, (bowl, plate)


def cut_closure(source):
    handles = tuple(source.handle('face', face) for face in source.faces) + tuple(
        source.handle('sheet', sheet) for sheet in source.sheets)
    _step('before extract')
    closure = owner.extract_model_closure(source, handles, include_features=False)
    _step('after extract')
    # Authenticate the ORIGINAL extraction before the owner composes its cut.
    _step('before prequalification')
    qualification = qualify_model_closure_source_publication(source, closure)
    _step('after prequalification')
    work = closure.working_model
    initial_edges = set(work.edges)
    operands = tuple(work.handle('face', face) for face in work.faces)
    _step('before plan')
    plan = owner.plan_intersections(work, operands, policy='connect')
    _step('after plan')
    _step('before apply')
    application = owner.apply_intersections(work, plan, policy='connect')
    _step('after apply')
    physical = {edge.id for edge in application.joint_edges}
    auxiliary = {edge for edge in set(work.edges)-initial_edges-physical
                 if len(work.faces_using_edge(edge)) == 2}
    assert len(work.faces) == 4
    assert sum(isinstance(face.surface, NurbsSurface) for face in work.faces.values()) == 2
    assert sum(isinstance(face.surface, owner.Plane) for face in work.faces.values()) == 2
    assert physical and auxiliary and not physical & auxiliary
    for edge in physical:
        assert len(work.faces_using_edge(edge)) == 4
    for edge in auxiliary:
        assert all(isinstance(work.faces[face].surface, NurbsSurface)
                   for face in work.faces_using_edge(edge))
    for original in source.faces:
        initial = closure.source_to_work[source.handle('face', original)]
        children = work.resolve_ref(EntityRef('face', initial.id))
        assert len(children) == 2
        for child in children:
            assert work.faces[child.id].metadata == source.faces[original].metadata
    assert not work.validate_topology()
    return closure, physical, auxiliary, qualification


def request(work, cancellation_check=None):
    _step('before policy selection')
    automation, selection = select_curvilinear_recovery_policy(work,
        requested_strategy='quad_first', order='linear', certification_mode='interactive',
        geometry_approximation_error=.0025, explicit_controls=False,
        cancellation_check=cancellation_check)
    _step('after policy selection')
    assert selection is not None
    _step('before generate')
    result = generate_automatic_mesh_result(work, automation=automation,
        curvilinear_recovery_selection=selection, target_size=.25,
        strategy='quad_first', order='linear', certification_mode='interactive',
        geometry_approximation_error=.0025, qualified_s3=True,
        structural_preparation=False, mutation_policy='working_copy', recombine=True,
        native_backend='python', cancellation_check=cancellation_check)
    _step('after generate')
    return result


def test_bounded_crossing_public_cancellation_preserves_original_and_cut_source():
    source, _parents = crossing_source()
    closure, _physical, _auxiliary, _qualification = cut_closure(source)
    before = owner.to_dict(source), owner.to_dict(closure.working_model)
    class Cancelled(BaseException):
        pass
    cancelled = Cancelled()
    def cancel(_phase):
        raise cancelled
    with pytest.raises(Cancelled) as caught:
        request(closure.working_model, cancel)
    assert caught.value is cancelled
    assert (owner.to_dict(source), owner.to_dict(closure.working_model)) == before
