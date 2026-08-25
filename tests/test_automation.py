from __future__ import annotations

from dataclasses import replace

import pytest

import anygeometry as ag
from anygeometry.automation import AutomationError
from anygeometry.automation.types import handle_to_dict

from anymesher.automation import (
    MeshAutomationSession,
    MeshCommand,
    MeshCommandBatch,
    automation_json_schema,
    automation_loads,
    describe_capabilities,
    tool_catalog,
)
from anymesher.serialize import mesh_to_dict


def _model() -> ag.GeometryModel:
    model = ag.GeometryModel()
    points = model.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    model.add_face(model.add_polyline(points, close=True))
    return model


def _batch(session: MeshAutomationSession, *commands: MeshCommand) -> MeshCommandBatch:
    return MeshCommandBatch(
        1,
        "request",
        session.session_id,
        session.geometry.model_id,
        session.geometry.revision,
        session.state_revision,
        commands,
    )


def test_provider_neutral_schema_and_strict_json() -> None:
    schema = automation_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert describe_capabilities()["provider_neutral"] is True
    assert describe_capabilities()["natural_language_interpretation"] is False
    assert {item["name"] for item in tool_catalog()} == {
        "mesh_capabilities",
        "mesh_session_summary",
        "select_geometry",
        "describe_geometry",
        "query_mesh",
        "plan_mesh",
        "apply_mesh",
    }
    with pytest.raises(AutomationError, match="duplicate JSON key"):
        automation_loads('{"protocol_version":1,"request_id":"x","tool":"mesh_capabilities","tool":"query_mesh","arguments":{}}')
    with pytest.raises(AutomationError, match="non-finite"):
        automation_loads('{"protocol_version":1,"request_id":"x","tool":"mesh_capabilities","arguments":{"x":NaN}}')


def test_plan_generate_apply_is_atomic_and_source_preserving() -> None:
    model = _model()
    revision = model.revision
    session = MeshAutomationSession(model)
    plan = session.plan(
        _batch(
            session,
            MeshCommand(
                "settings",
                "configure",
                {
                    "target_size": {"value": 500, "unit": "mm"},
                    "strategy": "mapped",
                },
            ),
            MeshCommand("generate", "generate", {}),
        )
    )
    assert model.revision == revision
    assert session.summary()["state_revision"] == 0
    assert plan.candidate_mesh_digest
    assert plan.candidate_summary["nodes"] > 0

    result = session.apply(plan)
    assert result.state_revision_after == 1
    assert model.revision == revision
    assert session.summary()["mesh"]["stale"] is False
    with pytest.raises(AutomationError, match="already been applied"):
        session.apply(plan)

    snapshot = session.mesh_snapshot()
    assert snapshot is not None
    expected = mesh_to_dict(snapshot)
    snapshot.nodes.clear()
    assert mesh_to_dict(session.mesh_snapshot()) == expected


def test_tamper_stale_failure_and_output_failure_leave_state_unchanged() -> None:
    session = MeshAutomationSession(_model())
    plan = session.plan(
        _batch(
            session,
            MeshCommand("settings", "configure", {"target_size": {"value": 0.5, "unit": "m"}}),
            MeshCommand("generate", "generate", {}),
        )
    )
    with pytest.raises(AutomationError, match="digest"):
        session.apply(replace(plan, controls_digest="0" * 64))
    assert session.state_revision == 0

    def fail_output(_mesh):
        raise OSError("disk full")

    with pytest.raises(AutomationError, match="disk full"):
        session.apply(plan, publisher=fail_output)
    assert session.state_revision == 0
    assert session.mesh_snapshot() is None


def test_scope_seed_refinement_queries_and_bounded_undo_redo() -> None:
    model = _model()
    session = MeshAutomationSession(model)
    edge = model.handle("edge", min(model.edges))
    plan = session.plan(
        _batch(
            session,
            MeshCommand("settings", "configure", {"target_size": {"value": 0.5, "unit": "m"}}),
            MeshCommand(
                "seed",
                "set_edge_divisions",
                {"targets": {"handles": [handle_to_dict(edge)]}, "divisions": 4},
            ),
            MeshCommand(
                "refine",
                "upsert_refinement",
                {
                    "name": "near_edge",
                    "target": {"handles": [handle_to_dict(edge)]},
                    "size": {"value": 250, "unit": "mm"},
                    "radius": {"value": 0.2, "unit": "m"},
                },
            ),
            MeshCommand("generate", "generate", {}),
        )
    )
    session.apply(plan)
    first_digest = session.summary()["mesh"]["mesh_digest"]
    nodes = session.query("nodes", {"page_size": 2})
    assert len(nodes["items"]) == 2
    assert nodes["next_cursor"]
    assert session.query("nodes", {"page_size": 1000, "cursor": nodes["next_cursor"]})["items"]
    association = session.query("associations", {"handle": handle_to_dict(edge)})
    assert association["nodes"]

    change = session.plan(
        _batch(
            session,
            MeshCommand("settings", "configure", {"target_size": {"value": 0.25, "unit": "m"}}),
        )
    )
    session.apply(change)
    assert session.summary()["mesh"]["stale"] is True
    undo = session.plan(_batch(session, MeshCommand("undo", "undo", {})))
    session.apply(undo)
    assert session.summary()["mesh"]["mesh_digest"] == first_digest
    assert session.summary()["mesh"]["stale"] is False
    redo = session.plan(_batch(session, MeshCommand("redo", "redo", {})))
    session.apply(redo)
    assert session.summary()["mesh"]["stale"] is True


def test_wrong_session_units_and_raw_mesh_edits_are_rejected() -> None:
    session = MeshAutomationSession(_model())
    with pytest.raises(AutomationError, match="another session"):
        session.plan(
            MeshCommandBatch(
                1,
                "bad",
                "00000000-0000-0000-0000-000000000000",
                session.geometry.model_id,
                session.geometry.revision,
                session.state_revision,
                (MeshCommand("generate", "generate", {}),),
            )
        )
    with pytest.raises(AutomationError, match="supported length"):
        session.plan(
            _batch(
                session,
                MeshCommand("settings", "configure", {"target_size": {"value": 1, "unit": "kg"}}),
            )
        )
    with pytest.raises(AutomationError, match="unsupported mesh command"):
        session.plan(_batch(session, MeshCommand("move", "move_node", {"id": 1})))
