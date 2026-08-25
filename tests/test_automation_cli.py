from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import anygeometry as ag
from anygeometry.serialization import write_geometry


def test_jsonl_cli_keeps_stdout_machine_readable(tmp_path: Path) -> None:
    model = ag.GeometryModel()
    points = model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    model.add_face(model.add_polyline(points, close=True))
    geometry_path = tmp_path / "geometry.json"
    write_geometry(geometry_path, model)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "anymesher",
            "automation",
            "--geometry",
            str(geometry_path),
        ],
        input=(
            '{"protocol_version":1,"request_id":"cap","tool":"mesh_capabilities","arguments":{}}\n'
            '{"protocol_version":1,"request_id":"bad","tool":"mesh_capabilities","arguments":{"path":"forbidden"}}\n'
        ),
        text=True,
        capture_output=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    rows = [json.loads(line) for line in process.stdout.splitlines()]
    assert [row["request_id"] for row in rows] == ["cap", "bad"]
    assert rows[0]["ok"] is True
    assert rows[0]["result"]["provider_neutral"] is True
    assert rows[1]["ok"] is False
    assert rows[1]["error"]["code"] == "UNKNOWN_FIELD"
    assert process.stderr == ""
