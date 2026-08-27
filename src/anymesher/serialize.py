"""JSON round-tripping for the neutral mesh.

This is not an interchange format.  A ``.fem`` or ``.inp`` file is written by
ANYfileio, which knows what other tools expect; this is the mesh container
written out as itself, so a command-line run can hand a mesh to the next step and
an editor can save one.

The association fields are part of the round trip.  A mesh whose node and element
groups were dropped on save would reload as a bag of coordinates, and every
restraint or load named against geometry would have nothing to resolve against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .errors import MeshError
from .mesh import Coupling, Mesh

__all__ = ["load_mesh", "mesh_from_dict", "mesh_to_dict", "save_mesh"]

FORMAT = "anymesher.mesh"
FORMAT_VERSION = 3


def _json_value(value: Any, path: str) -> Any:
    """Return deterministic JSON data or fail rather than stringify objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        made = float(value)
        if not np.isfinite(made):
            raise MeshError(f"{path} contains a non-finite number")
        return made
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        made: Dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            name = str(key)
            made[name] = _json_value(value[key], f"{path}.{name}")
        return made
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path)
    raise MeshError(
        f"{path} contains unsupported {type(value).__name__}; "
        "mesh audit data must be JSON-safe"
    )


def mesh_to_dict(mesh: Mesh) -> Dict[str, Any]:
    """Return a JSON-safe mapping of a mesh."""

    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "geometry_model_id": (
            None if mesh.geometry_model_id is None else str(mesh.geometry_model_id)
        ),
        "geometry_revision": mesh.geometry_revision,
        "order": mesh.order,
        "automatic_intersections": int(mesh.automatic_intersections),
        "declared_plate_junction_edges": [
            [int(first), int(second)]
            for first, second in mesh.declared_plate_junction_edges
        ],
        "automatic_beam_connections": int(mesh.automatic_beam_connections),
        "automatic_shell_connections": int(mesh.automatic_shell_connections),
        "nodes": {
            str(node_id): [float(value) for value in position]
            for node_id, position in sorted(mesh.nodes.items())
        },
        "quads": {str(k): list(map(int, v)) for k, v in sorted(mesh.quads.items())},
        "tris": {str(k): list(map(int, v)) for k, v in sorted(mesh.tris.items())},
        "beams": {str(k): list(map(int, v)) for k, v in sorted(mesh.beams.items())},
        "couplings": {
            str(element_id): {
                "beam_node": int(coupling.beam_node),
                "plate_nodes": [int(node) for node in coupling.plate_nodes],
                "weights": [float(weight) for weight in coupling.weights],
                "eccentricity": [float(value) for value in coupling.eccentricity],
            }
            for element_id, coupling in sorted(mesh.couplings.items())
        },
        "node_of_vertex": {str(k): int(v) for k, v in sorted(mesh.node_of_vertex.items())},
        "nodes_of_edge": {str(k): list(map(int, v)) for k, v in sorted(mesh.nodes_of_edge.items())},
        "offset_nodes_of_edge": {
            str(k): list(map(int, v)) for k, v in sorted(mesh.offset_nodes_of_edge.items())
        },
        "elements_of_face": {
            str(k): list(map(int, v)) for k, v in sorted(mesh.elements_of_face.items())
        },
        "elements_of_edge": {
            str(k): list(map(int, v)) for k, v in sorted(mesh.elements_of_edge.items())
        },
        "elements_of_sheet": {
            str(k): list(map(int, v)) for k, v in sorted(mesh.elements_of_sheet.items())
        },
        "elements_of_member": {
            str(k): list(map(int, v)) for k, v in sorted(mesh.elements_of_member.items())
        },
        "nodes_of_member": {
            str(k): list(map(int, v)) for k, v in sorted(mesh.nodes_of_member.items())
        },
        "activity": {
            str(k): float(v) for k, v in sorted(mesh.activity.items())
        },
        "grid_of_face": {
            str(k): [[int(value) for value in row] for row in np.asarray(v).tolist()]
            for k, v in sorted(mesh.grid_of_face.items())
        },
        "block_grids_of_face": {
            str(face_id): [
                [[int(value) for value in row] for row in np.asarray(grid).tolist()]
                for grid in grids
            ]
            for face_id, grids in sorted(mesh.block_grids_of_face.items())
        },
        "thickness_of_face": {
            str(k): float(v) for k, v in sorted(mesh.thickness_of_face.items())
        },
        "structural_preparation": _json_value(
            mesh.structural_preparation, "structural_preparation"
        ),
        "hybrid_diagnostics": _json_value(
            mesh.hybrid_diagnostics, "hybrid_diagnostics"
        ),
    }


def mesh_from_dict(data: Mapping[str, Any]) -> Mesh:
    """Rebuild a mesh from :func:`mesh_to_dict` output.

    The seeding is not restored: it records how the mesh was asked for rather
    than what it is, and a reloaded mesh that claimed a seeding it could not
    reproduce would be worse than one that admits it has none.
    """

    if data.get("format") != FORMAT:
        raise MeshError(f"not an {FORMAT} document: format={data.get('format')!r}")
    version = int(data.get("version", 0))
    if version not in (1, 2, FORMAT_VERSION):
        raise MeshError(
            f"unsupported {FORMAT} version {version}; this build reads 1-{FORMAT_VERSION}"
        )

    geometry_revision = data.get("geometry_revision")
    if geometry_revision is not None:
        geometry_revision = int(geometry_revision)
        if geometry_revision < 0:
            raise MeshError("geometry_revision must be non-negative")

    declared_plate_junction_edges = tuple(
        tuple(int(node) for node in edge)
        for edge in data.get("declared_plate_junction_edges", ())
    )
    if any(
        len(edge) != 2 or edge[0] == edge[1]
        for edge in declared_plate_junction_edges
    ):
        raise MeshError("declared_plate_junction_edges must contain node-ID pairs")

    mesh = Mesh(
        geometry_model_id=data.get("geometry_model_id"),
        geometry_revision=geometry_revision,
        order=str(data.get("order", "linear")),
        automatic_intersections=int(data.get("automatic_intersections", 0)),
        declared_plate_junction_edges=tuple(
            sorted(
                {
                    (min(first, second), max(first, second))
                    for first, second in declared_plate_junction_edges
                }
            )
        ),
        automatic_beam_connections=int(data.get("automatic_beam_connections", 0)),
        automatic_shell_connections=int(data.get("automatic_shell_connections", 0)),
        structural_preparation=dict(
            _json_value(
                data.get("structural_preparation", {}),
                "structural_preparation",
            )
        ),
        hybrid_diagnostics=dict(
            _json_value(data.get("hybrid_diagnostics", {}), "hybrid_diagnostics")
        ),
    )
    for node_id, position in data.get("nodes", {}).items():
        mesh.nodes[int(node_id)] = np.asarray(position, dtype=float)
    for name in ("quads", "tris", "beams"):
        target = getattr(mesh, name)
        for element_id, connectivity in data.get(name, {}).items():
            target[int(element_id)] = tuple(int(node) for node in connectivity)
    for element_id, record in data.get("couplings", {}).items():
        mesh.couplings[int(element_id)] = Coupling(
            beam_node=int(record["beam_node"]),
            plate_nodes=tuple(int(node) for node in record["plate_nodes"]),
            weights=tuple(float(weight) for weight in record["weights"]),
            eccentricity=tuple(float(value) for value in record["eccentricity"]),  # type: ignore[arg-type]
        )
    for vertex_id, node_id in data.get("node_of_vertex", {}).items():
        mesh.node_of_vertex[int(vertex_id)] = int(node_id)
    for name in (
        "nodes_of_edge",
        "offset_nodes_of_edge",
        "elements_of_face",
        "elements_of_edge",
        "elements_of_sheet",
        "elements_of_member",
        "nodes_of_member",
    ):
        target = getattr(mesh, name)
        for key, values in data.get(name, {}).items():
            target[int(key)] = [int(value) for value in values]
    for element_id, value in data.get("activity", {}).items():
        made = float(value)
        if not np.isfinite(made) or made < 0.0 or made > 1.0:
            raise MeshError(f"invalid activity for element {element_id}: {value!r}")
        mesh.activity[int(element_id)] = made
    for face_id, grid in data.get("grid_of_face", {}).items():
        mesh.grid_of_face[int(face_id)] = np.asarray(grid, dtype=int)
    for face_id, grids in data.get("block_grids_of_face", {}).items():
        mesh.block_grids_of_face[int(face_id)] = tuple(
            np.asarray(grid, dtype=int) for grid in grids
        )
    for face_id, thickness in data.get("thickness_of_face", {}).items():
        mesh.thickness_of_face[int(face_id)] = float(thickness)

    missing = sorted(
        {
            node
            for connectivity in list(mesh.quads.values())
            + list(mesh.tris.values())
            + list(mesh.beams.values())
            for node in connectivity
            if node not in mesh.nodes
        }
    )
    if missing:
        raise MeshError(f"mesh references undefined node(s) {missing[:8]}")
    return mesh


def save_mesh(path: str | Path, mesh: Mesh, *, overwrite: bool = False) -> Path:
    """Write a mesh to a JSON file."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(mesh_to_dict(mesh), indent=1) + "\n", encoding="utf-8")
    return destination


def load_mesh(path: str | Path) -> Mesh:
    """Read a mesh from a JSON file."""

    return mesh_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
