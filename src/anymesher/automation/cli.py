"""JSON Lines process transport for :mod:`anymesher.automation`."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from anygeometry.automation import AutomationError, AutomationResponse, PROTOCOL_VERSION
from anygeometry.serialization import read_geometry

from ..mesh import Mesh
from ..serialize import load_mesh, mesh_to_dict
from .schema import automation_dumps, automation_loads
from .session import MeshAutomationSession, dispatch_tool


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "automation", help="run a provider-neutral JSON Lines mesh command session"
    )
    parser.add_argument("--geometry", required=True, help="ANYgeometry JSON document")
    parser.add_argument("--mesh", help="optional matching ANYmesher JSON document")
    parser.add_argument("--output", help="host-owned current-mesh output path")
    parser.add_argument("--overwrite", action="store_true", help="replace a pre-existing output")


class _AtomicPublisher:
    def __init__(self, path: str | None, overwrite: bool) -> None:
        self.path = None if path is None else Path(path)
        self.overwrite = bool(overwrite)
        self.owned = False
        if self.path is not None and self.path.exists() and not self.overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {self.path}")

    def __call__(self, mesh: Mesh | None) -> None:
        if self.path is None or mesh is None:
            return
        destination = self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        if partial.exists():
            raise FileExistsError(f"refusing existing partial output: {partial}")
        text = json.dumps(mesh_to_dict(mesh), indent=1) + "\n"
        try:
            with partial.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, destination)
            self.owned = True
        except Exception:
            if partial.exists():
                partial.unlink()
            raise


def run(args: Any) -> int:
    geometry = read_geometry(args.geometry)
    mesh = None if args.mesh is None else load_mesh(args.mesh)
    session = MeshAutomationSession(geometry, mesh)
    publisher = _AtomicPublisher(args.output, args.overwrite)
    for raw in sys.stdin:
        request_id = "unknown"
        try:
            request = automation_loads(raw.rstrip("\r\n"))
            request_id = str(request["request_id"])
            response = dispatch_tool(session, request, publisher=publisher)
        except AutomationError as error:
            response = AutomationResponse(
                PROTOCOL_VERSION, request_id, False, error=error
            ).to_dict()
        sys.stdout.write(automation_dumps(response) + "\n")
        sys.stdout.flush()
    return 0


__all__ = ["add_parser", "run"]
