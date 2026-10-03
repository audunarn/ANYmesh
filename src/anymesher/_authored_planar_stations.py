"""Owner-backed original-UV station preflight for an opt-in planar route."""

from dataclasses import dataclass
from fractions import Fraction

from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError


@dataclass(frozen=True)
class AuthoredPlanarStationPlan:
    authored_face: int
    # Owner receipts preserve exact rational UV/XYZ and source parameters.
    exterior_receipts: tuple[object, ...]
    interior_receipts: tuple[object, ...]
    material_receipts: tuple[object, ...]
    # Exact CURRENT material trace in the original chart, not source ancestry.
    node_material_uv: tuple[tuple[int, tuple[Fraction, Fraction]], ...]
    vertex_preimages: object

    @property
    def publication_qualified(self) -> bool:
        return False


def plan_authored_planar_stations(
    geometry, correspondence, mesh, registry: GlobalEdgeBoundaryRegistry,
    cancellation_check=None,
) -> AuthoredPlanarStationPlan:
    """Validate registered exterior IDs/XYZ before any native root work."""
    try:
        from anygeometry import (
            query_prepared_authored_boundary_stations,
            query_prepared_authored_internal_stations,
            query_prepared_authored_material_stations,
            query_prepared_vertex_preimages,
            validate_prepared_authored_boundary_correspondence_binding,
            validate_prepared_authored_boundary_station_coordinates,
            validate_prepared_authored_internal_station_coordinates,
            validate_prepared_authored_material_station_coordinates,
            validate_prepared_vertex_preimages_binding,
        )
    except ImportError as error:
        raise MeshError("authored planar station capability is unavailable") from error
    if not isinstance(registry, GlobalEdgeBoundaryRegistry):
        raise MeshError("authored planar stations need a global edge registry")
    validate_prepared_authored_boundary_correspondence_binding(
        geometry, correspondence, cancellation_check=cancellation_check,
    )
    registry.view.assert_current(geometry)
    exterior = tuple(dict.fromkeys(
        int(edge_id)
        for loop in correspondence.exterior_loops
        for _source, _forward, current in loop
        for edge_id in current
    ))
    interior = tuple(int(edge_id) for edge_id, _uses in correspondence.interior_incidence)
    if len(interior) != len(set(interior)) or set(exterior) & set(interior):
        raise MeshError("authored planar edges have ambiguous current incidence")
    if not exterior:
        raise MeshError("authored planar root has no qualified exterior edges")
    exterior_receipts, interior_receipts, material_receipts = [], [], []
    material_uv_by_node = {}
    for edge_id in (*exterior, *interior):
        entries = registry.entries(edge_id)
        if (len(entries) < 2 or entries[0].key.parameter != 0.0
                or entries[-1].key.parameter != 1.0):
            raise MeshError(f"authored exterior edge {edge_id} lacks complete stations")
        if any(entry.node_id is None or entry.node_id not in mesh.nodes
               for entry in entries):
            raise MeshError(f"authored exterior edge {edge_id} lacks published node IDs")
        coordinates = [mesh.nodes[entry.node_id] for entry in entries]
        parameters = [entry.key.parameter for entry in entries]
        if edge_id in exterior:
            receipt = query_prepared_authored_boundary_stations(
                geometry, correspondence, edge_id, parameters,
                cancellation_check=cancellation_check,
            )
            validate_prepared_authored_boundary_station_coordinates(
                geometry, receipt, coordinates,
                cancellation_check=cancellation_check,
            )
            exterior_receipts.append(receipt)
        else:
            receipt = query_prepared_authored_internal_stations(
                geometry, correspondence, edge_id, parameters,
                cancellation_check=cancellation_check,
            )
            validate_prepared_authored_internal_station_coordinates(
                geometry, receipt, coordinates,
                cancellation_check=cancellation_check,
            )
            if receipt.endpoint_ids != (geometry.edges[edge_id].start,
                                        geometry.edges[edge_id].end):
                raise MeshError(f"authored interior edge {edge_id} changed endpoints")
            interior_receipts.append(receipt)
        material = query_prepared_authored_material_stations(
            geometry, correspondence, edge_id, parameters,
            cancellation_check=cancellation_check,
        )
        validate_prepared_authored_material_station_coordinates(
            geometry, material, coordinates,
            cancellation_check=cancellation_check,
        )
        if material.endpoint_ids != (geometry.edges[edge_id].start,
                                     geometry.edges[edge_id].end):
            raise MeshError(f"authored material edge {edge_id} changed endpoints")
        for entry, pair in zip(entries, material.authored_uv):
            uv = tuple(Fraction(*value) for value in pair)
            previous = material_uv_by_node.setdefault(entry.node_id, uv)
            if previous != uv:
                raise MeshError(
                    f"authored current material node {entry.node_id} has inconsistent UV"
                )
        material_receipts.append(material)
    vertices = query_prepared_vertex_preimages(
        geometry, cancellation_check=cancellation_check,
    )
    validate_prepared_vertex_preimages_binding(
        geometry, vertices, cancellation_check=cancellation_check,
    )
    validate_prepared_authored_boundary_correspondence_binding(
        geometry, correspondence, cancellation_check=cancellation_check,
    )
    registry.view.assert_current(geometry)
    return AuthoredPlanarStationPlan(
        int(correspondence.authored_definition.face_id),
        tuple(exterior_receipts), tuple(interior_receipts),
        tuple(material_receipts), tuple(sorted(material_uv_by_node.items())),
        vertices,
    )
