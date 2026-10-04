"""Private exact linear shell associations for one complete authored component.

This proves staged CURRENT cell/edge/vertex/Sheet/FaceUse buckets and literal
material partition. It neither remaps source project references nor admits a
mesh for publication or analysis.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import numpy as np

from ._authored_component_binding import (
    BoundAuthoredSheetJointComponent, bind_authored_sheet_joint_component,
)
from ._authored_planar_stations import plan_authored_planar_stations
from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError


@dataclass(frozen=True)
class AuthoredComponentCellAssociations:
    current_face_cells: tuple[tuple[int, tuple[int, ...]], ...]
    sheet_cells: tuple[tuple[int, tuple[int, ...]], ...]
    face_use_cells: tuple[tuple[int, tuple[int, ...]], ...]
    edge_chains: tuple[tuple[int, tuple[int, ...]], ...]
    vertex_nodes: tuple[tuple[int, int], ...]
    joint_chains: tuple[tuple[int, tuple[int, ...]], ...]
    cell_current_faces: tuple[tuple[int, int], ...]

    @property
    def publication_qualified(self) -> bool:
        return False


def validate_authored_component_cells(
    geometry, component: BoundAuthoredSheetJointComponent, mesh,
    registry: GlobalEdgeBoundaryRegistry, cell_current_faces, *,
    created_material_uv_by_root=None,
    cancellation_check=None,
) -> AuthoredComponentCellAssociations:
    """Prove exact source-only CURRENT buckets for linear planar shell cells."""
    try:
        from anygeometry import validate_prepared_authored_face_partition
    except ImportError as error:
        raise MeshError("authored component partition capability is unavailable") from error
    if not isinstance(component, BoundAuthoredSheetJointComponent):
        raise MeshError("authored component cells need a bound whole component")
    current = bind_authored_sheet_joint_component(
        geometry, component.joint_edge_id, component.boundary_correspondences,
        component.authored_face_ids, cancellation_check=cancellation_check,
    )
    if (current.authored_face_ids != component.authored_face_ids
            or current.current_face_ids != component.current_face_ids
            or current.occurrence_correspondence != component.occurrence_correspondence
            or current.owner_receipt != component.owner_receipt):
        raise MeshError("authored component cells bind changed owner evidence")
    if not isinstance(registry, GlobalEdgeBoundaryRegistry):
        raise MeshError("authored component cells need a global edge registry")
    registry.view.assert_current(geometry)
    if (mesh.geometry_model_id != geometry.model_id
            or mesh.geometry_revision != geometry.revision
            or mesh.order != "linear" or mesh.beams or mesh.couplings):
        raise MeshError("authored component cells need current linear shell identity")
    if (mesh.elements_of_edge or mesh.elements_of_member or mesh.nodes_of_member
            or mesh.offset_nodes_of_edge or mesh.grid_of_face
            or mesh.block_grids_of_face or mesh.structural_preparation
            or mesh.hybrid_diagnostics or mesh.automatic_beam_connections
            or mesh.automatic_shell_connections or mesh.automatic_intersections
            or mesh.seeding is not None or mesh.thickness_of_face or mesh.activity):
        raise MeshError("authored component cells contain unsupported associations")
    shells = {**mesh.tris, **mesh.quads}
    if not shells or len(shells) != len(mesh.tris) + len(mesh.quads):
        raise MeshError("authored component cells need unique shell IDs")
    if (set(shells) != set(cell_current_faces)
            or any(isinstance(cell, bool) or not isinstance(cell, Integral)
                   or isinstance(face, bool) or not isinstance(face, Integral)
                   or int(face) not in current.current_face_ids
                   for cell, face in cell_current_faces.items())):
        raise MeshError("authored component cells need one current face per shell")
    face_cells = {face: [] for face in current.current_face_ids}
    shell_segments = set()
    face_root = {int(face): int(correspondence.authored_definition.face_id)
                 for correspondence in current.boundary_correspondences
                 for face in correspondence.descendants}
    segment_roots = defaultdict(set)
    active_nodes = set()
    for cell_id, connection in shells.items():
        corners = 3 if cell_id in mesh.tris else 4
        if len(connection) != corners or len(set(connection)) != corners:
            raise MeshError("authored component cells need linear T3/Q4 corners")
        if any(node not in mesh.nodes for node in connection):
            raise MeshError("authored component cell has an unpublished node")
        face_cells[int(cell_current_faces[cell_id])].append(int(cell_id))
        active_nodes.update(int(node) for node in connection)
        for index in range(corners):
            a, b = int(connection[index]), int(connection[(index + 1) % corners])
            edge = (min(a, b), max(a, b))
            shell_segments.add(edge)
            segment_roots[edge].add(face_root[int(cell_current_faces[cell_id])])
    if any(not ids for ids in face_cells.values()):
        raise MeshError("authored component has a current face without cells")
    expected_faces = {face: sorted(ids) for face, ids in face_cells.items()}
    if mesh.elements_of_face != expected_faces:
        raise MeshError("authored component current face cell buckets changed")

    uses = {int(row["id"]): row
            for row in current.owner_receipt.current_records["face_uses"]}
    use_cells = {}
    sheet_cells = defaultdict(set)
    for sheet, _root, _original_use, current_uses in current.occurrence_correspondence:
        for use_id in current_uses:
            use = uses.get(int(use_id))
            if use is None or int(use["sheet_id"]) != int(sheet):
                raise MeshError("authored component current FaceUse meaning changed")
            ids = tuple(expected_faces[int(use["face_id"])])
            if int(use_id) in use_cells:
                raise MeshError("authored component repeated a current FaceUse")
            use_cells[int(use_id)] = ids
            sheet_cells[int(sheet)].update(ids)
    if set(use_cells) != set(uses) or set(sheet_cells) != set(current.sheet_ids):
        raise MeshError("authored component omitted current Sheet occurrences")
    expected_sheets = {sheet: sorted(ids) for sheet, ids in sheet_cells.items()}
    if mesh.elements_of_sheet != expected_sheets:
        raise MeshError("authored component current Sheet cell buckets changed")

    expected_edges = {
        int(edge) for correspondence in current.boundary_correspondences
        for loop in correspondence.exterior_loops
        for _source, _forward, edges in loop for edge in edges
    } | {
        int(edge) for correspondence in current.boundary_correspondences
        for edge, _uses in correspondence.interior_incidence
    }
    if set(mesh.nodes_of_edge) != expected_edges:
        raise MeshError("authored component lacks complete current edge chains")
    edge_chains = {}
    for edge in sorted(expected_edges):
        entries = registry.entries(edge)
        chain = tuple(entry.node_id for entry in entries)
        if (len(chain) < 2 or entries[0].key.parameter != 0.0
                or entries[-1].key.parameter != 1.0
                or any(node is None or node not in mesh.nodes for node in chain)
                or mesh.nodes_of_edge[edge] != list(chain)):
            raise MeshError(f"authored component edge {edge} has changed station order")
        if any((min(a, b), max(a, b)) not in shell_segments
               for a, b in zip(chain, chain[1:])):
            raise MeshError(f"authored component edge {edge} is not a shell chain")
        edge_chains[edge] = chain
    expected_vertices = {int(row["id"])
                         for row in current.owner_receipt.current_records["vertices"]}
    if set(mesh.node_of_vertex) != expected_vertices:
        raise MeshError("authored component lacks complete current vertex nodes")
    for vertex, node in mesh.node_of_vertex.items():
        if node not in active_nodes or node not in mesh.nodes:
            raise MeshError("authored component vertex lacks an active shell node")
        xyz = np.asarray(mesh.nodes[node], dtype=float)
        if (xyz.shape != (3,) or not np.all(np.isfinite(xyz))
                or np.linalg.norm(xyz - geometry.vertex_position(vertex))
                > registry.view.effective_length()):
            raise MeshError("authored component vertex coordinate changed")
    if set(mesh.nodes) != active_nodes:
        raise MeshError("authored component has an unassociated node")
    joint_chains = tuple((edge, edge_chains[edge])
                         for edge in current.owner_receipt.joint_edge_ids)
    joint_segments = {
        (min(a, b), max(a, b)) for _edge, chain in joint_chains
        for a, b in zip(chain, chain[1:])
    }
    if (len(mesh.declared_plate_junction_edges) != len(joint_segments)
            or set(mesh.declared_plate_junction_edges) != joint_segments):
        raise MeshError("authored component joint declaration changed")
    if any(segment_roots[edge] != set(current.authored_face_ids)
           for edge in joint_segments):
        raise MeshError("authored component joint lacks shell incidence from both roots")

    if created_material_uv_by_root is None:
        created_material_uv_by_root = {
            correspondence.authored_definition.face_id: {}
            for correspondence in current.boundary_correspondences
        }
    if (not isinstance(created_material_uv_by_root, Mapping)
            or set(created_material_uv_by_root) != set(current.authored_face_ids)
            or any(isinstance(root, bool) or not isinstance(root, Integral)
                   for root in created_material_uv_by_root)
            or any(not isinstance(values, Mapping)
                   for values in created_material_uv_by_root.values())):
        raise MeshError("authored component created UV needs every original root")
    created_ids = set()

    for correspondence in current.boundary_correspondences:
        plan = plan_authored_planar_stations(
            geometry, correspondence, mesh, registry,
            cancellation_check=cancellation_check,
        )
        uv = dict(plan.node_material_uv)
        root = int(correspondence.authored_definition.face_id)
        created = created_material_uv_by_root[root]
        root_used = {
            int(node) for face in correspondence.descendants
            for cell in expected_faces[face] for node in shells[cell]
        }
        if set(created) != root_used - set(uv):
            raise MeshError("authored component created UV does not match root nodes")
        if set(created) & created_ids:
            raise MeshError("authored component created node crosses original roots")
        created_ids.update(created)
        for node, pair in created.items():
            if (isinstance(node, bool) or not isinstance(node, Integral)
                    or node not in mesh.nodes or not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(type(value) is not Fraction for value in pair)):
                raise MeshError("authored component created UV needs exact node provenance")
        if created:
            from anygeometry import evaluate_prepared_authored_face

            nodes = tuple(sorted(created))
            expected_xyz = evaluate_prepared_authored_face(
                geometry, correspondence,
                [[float(value) for value in created[node]] for node in nodes],
                cancellation_check=cancellation_check,
            )
            actual_xyz = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
            if (actual_xyz.shape != expected_xyz.shape
                    or not np.all(np.isfinite(actual_xyz))
                    or np.any(np.linalg.norm(actual_xyz - expected_xyz, axis=1)
                              > registry.view.effective_length())):
                raise MeshError("authored component created node left owner support")
            uv.update(created)
        triangles_by_child = {}
        for face in correspondence.descendants:
            triangles = []
            for cell_id in expected_faces[face]:
                connection = shells[cell_id]
                if any(node not in uv for node in connection):
                    raise MeshError("authored component cell lacks exact material UV")
                points = tuple(uv[node] for node in connection)
                if len(points) == 3:
                    triangles.append(points)
                else:
                    triangles.extend(((points[0], points[1], points[2]),
                                      (points[0], points[2], points[3])))
            triangles_by_child[face] = tuple(triangles)
        validate_prepared_authored_face_partition(
            geometry, correspondence, triangles_by_child,
            cancellation_check=cancellation_check,
        )
    registry.view.assert_current(geometry)
    return AuthoredComponentCellAssociations(
        tuple((face, tuple(ids)) for face, ids in sorted(expected_faces.items())),
        tuple((sheet, tuple(ids)) for sheet, ids in sorted(expected_sheets.items())),
        tuple(sorted(use_cells.items())), tuple(sorted(edge_chains.items())),
        tuple(sorted((int(vertex), int(node)) for vertex, node in mesh.node_of_vertex.items())),
        joint_chains, tuple(sorted((int(cell), int(face))
                                   for cell, face in cell_current_faces.items())),
    )
