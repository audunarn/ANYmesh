"""Command line interface.

``python -m anymesher <command>``, or ``anymesher <command>`` once installed.
Every command takes ``--json`` for machine-readable output.

The commands cover the primitives and the inspection of a saved mesh. Geometry
documents are serialized by ANYgeometry; this mesher CLI intentionally accepts
mesh primitives and mesh files only. Applications mesh a shared GeometryModel
through the public API and keep project serialization in their owning package.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Sequence

from .backends import available_backends
from .errors import GeometryError, MeshError
from .mesh import Mesh
from .primitives import (
    PanelMeshConfig,
    StiffenedPanel,
    StiffenerCrossSection,
    beam_mesh,
    panel_edge_nodes,
    simple_panel_mesh,
    stiffened_panel_mesh,
)
from .quality import verify_mesh_quality
from .serialize import load_mesh, save_mesh

__all__ = ["main"]


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))


def _summary(mesh: Mesh) -> Dict[str, Any]:
    quality = verify_mesh_quality(mesh)
    return {
        "order": mesh.order,
        "nodes": mesh.num_nodes,
        "quads": len(mesh.quads),
        "tris": len(mesh.tris),
        "beams": len(mesh.beams),
        "couplings": len(mesh.couplings),
        "faces": sorted(mesh.elements_of_face),
        "edges": sorted(mesh.nodes_of_edge),
        "quality": quality.as_dict(),
    }


def _report(mesh: Mesh, args: argparse.Namespace, extra: Dict[str, Any] | None = None) -> int:
    summary = _summary(mesh)
    if extra:
        summary.update(extra)
    if args.output:
        save_mesh(args.output, mesh, overwrite=args.overwrite)
        summary["written"] = str(args.output)

    if args.json:
        _print_json(summary)
    else:
        print(f"{summary['order']} mesh: {summary['nodes']} nodes, {summary['quads']} quads, "
              f"{summary['tris']} tris, {summary['beams']} beams, {summary['couplings']} couplings")
        quality = summary["quality"]
        print(f"  aspect ratio    max {quality['max_aspect_ratio']:.3f}  "
              f"mean {quality['mean_aspect_ratio']:.3f}")
        print(f"  warp            max {quality['max_warp']:.5f}")
        for name, values in extra.items() if extra else ():
            print(f"  {name:<15} {values}")
        for warning in quality["warnings"]:
            print(f"  warning: {warning}")
        if summary.get("written"):
            print(f"  written to {summary['written']}")
    # A quality warning is advisory, so it does not change the exit code.
    return 0


def _command_backends(args: argparse.Namespace) -> int:
    names = list(available_backends())
    if args.json:
        _print_json({"backends": names})
        return 0
    print("mesh backends: " + ", ".join(names))
    print("  mapped  built in; structured grid per face, conformity by construction")
    print("  gmsh    needs ANYmesher[gmsh]; unstructured, planar faces, may leave triangles")
    return 0


def _command_panel(args: argparse.Namespace) -> int:
    panel = StiffenedPanel(
        length=args.length,
        width=args.width,
        plate_thickness=args.thickness,
        stiffener_type=args.stiffener_type,
        stiffener_spacing=args.spacing,
        stiffener_height=args.height,
        stiffener_web_thickness=args.web_thickness,
        stiffener_flange_width=args.flange_width,
        stiffener_flange_thickness=args.flange_thickness,
        num_stiffeners=args.stiffeners,
    )
    config = PanelMeshConfig(
        shell_num_divisions_x=args.divisions_x,
        shell_num_divisions_y=args.divisions_y,
        beam_num_divisions=args.beam_divisions,
        use_8node_shells=args.quadratic,
        align_mesh_to_stiffeners=args.align,
    )
    mesh = stiffened_panel_mesh(panel, config)
    section = StiffenerCrossSection.from_panel(panel)
    extra = {
        "edge_nodes": {name: len(nodes) for name, nodes in panel_edge_nodes(mesh).items()},
        "stiffener_section": {
            "area": section.area,
            "Iy": section.Iy,
            "Iz": section.Iz,
            "J": section.J,
        },
    }
    return _report(mesh, args, extra)


def _command_plate(args: argparse.Namespace) -> int:
    mesh = simple_panel_mesh(
        args.length, args.width, args.thickness, args.divisions_x, args.divisions_y, args.quadratic
    )
    return _report(mesh, args, {"edge_nodes": {n: len(v) for n, v in panel_edge_nodes(mesh).items()}})


def _command_beam(args: argparse.Namespace) -> int:
    return _report(beam_mesh(args.length, args.divisions), args)


def _command_quality(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.input)
    args.output = None
    return _report(mesh, args)


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", "-o", help="write the mesh to a JSON file")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anymesher", description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("backends", help="list mesh backends")

    panel = sub.add_parser("panel", help="mesh a rectangular stiffened panel")
    panel.add_argument("--length", type=float, required=True, help="panel length in m")
    panel.add_argument("--width", type=float, required=True, help="panel width in m")
    panel.add_argument("--thickness", type=float, required=True, help="plate thickness in m")
    panel.add_argument("--stiffeners", type=int, default=1)
    panel.add_argument("--spacing", type=float, default=0.0, help="0 divides the width evenly")
    panel.add_argument("--stiffener-type", default="T-bar")
    panel.add_argument("--height", type=float, default=0.0, help="web height in m")
    panel.add_argument("--web-thickness", type=float, default=0.0)
    panel.add_argument("--flange-width", type=float, default=0.0)
    panel.add_argument("--flange-thickness", type=float, default=0.0)
    panel.add_argument("--divisions-x", type=int, default=4)
    panel.add_argument("--divisions-y", type=int, default=4)
    panel.add_argument("--beam-divisions", type=int, default=1)
    panel.add_argument("--quadratic", action="store_true", help="8-node shells")
    panel.add_argument("--align", action="store_true", help="put a mesh line on every stiffener")
    _add_output_options(panel)

    plate = sub.add_parser("plate", help="mesh a flat unstiffened plate")
    plate.add_argument("--length", type=float, required=True)
    plate.add_argument("--width", type=float, required=True)
    plate.add_argument("--thickness", type=float, required=True)
    plate.add_argument("--divisions-x", type=int, default=4)
    plate.add_argument("--divisions-y", type=int, default=4)
    plate.add_argument("--quadratic", action="store_true")
    _add_output_options(plate)

    beam = sub.add_parser("beam", help="mesh a straight beam")
    beam.add_argument("--length", type=float, required=True)
    beam.add_argument("--divisions", type=int, default=10)
    _add_output_options(beam)

    quality = sub.add_parser("quality", help="report on a saved mesh")
    quality.add_argument("input")

    args = parser.parse_args(argv)
    handlers = {
        "backends": _command_backends,
        "panel": _command_panel,
        "plate": _command_plate,
        "beam": _command_beam,
        "quality": _command_quality,
    }
    try:
        return handlers[args.command](args)
    except (FileExistsError, FileNotFoundError, GeometryError, MeshError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
