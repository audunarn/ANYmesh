"""Owner-backed original-UV station preflight for an opt-in planar route.

Internal physical constraints are deliberately refused until ANYgeometry can
bind their current edge stations to original UV without XYZ inversion.
"""

from dataclasses import dataclass

from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError


@dataclass(frozen=True)
class AuthoredPlanarStationPlan:
    authored_face: int
    # Owner receipts preserve exact rational UV/XYZ and source parameters.
    exterior_receipts: tuple[object, ...]

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
            validate_prepared_authored_boundary_correspondence_binding,
            validate_prepared_authored_boundary_station_coordinates,
        )
    except ImportError as error:
        raise MeshError("authored planar station capability is unavailable") from error
    if not isinstance(registry, GlobalEdgeBoundaryRegistry):
        raise MeshError("authored planar stations need a global edge registry")
    validate_prepared_authored_boundary_correspondence_binding(
        geometry, correspondence, cancellation_check=cancellation_check,
    )
    registry.view.assert_current(geometry)
    if correspondence.interior_incidence:
        edge_id = int(correspondence.interior_incidence[0][0])
        raise MeshError(
            f"authored root interior edge {edge_id} needs owner original-UV station mapping"
        )
    edges = tuple(dict.fromkeys(
        int(edge_id)
        for loop in correspondence.exterior_loops
        for _source, _forward, current in loop
        for edge_id in current
    ))
    if not edges:
        raise MeshError("authored planar root has no qualified exterior edges")
    receipts = []
    for edge_id in edges:
        entries = registry.entries(edge_id)
        if (len(entries) < 2 or entries[0].key.parameter != 0.0
                or entries[-1].key.parameter != 1.0):
            raise MeshError(f"authored exterior edge {edge_id} lacks complete stations")
        if any(entry.node_id is None or entry.node_id not in mesh.nodes
               for entry in entries):
            raise MeshError(f"authored exterior edge {edge_id} lacks published node IDs")
        coordinates = [mesh.nodes[entry.node_id] for entry in entries]
        receipt = query_prepared_authored_boundary_stations(
            geometry, correspondence, edge_id,
            [entry.key.parameter for entry in entries],
            cancellation_check=cancellation_check,
        )
        validate_prepared_authored_boundary_station_coordinates(
            geometry, receipt, coordinates,
            cancellation_check=cancellation_check,
        )
        receipts.append(receipt)
    validate_prepared_authored_boundary_correspondence_binding(
        geometry, correspondence, cancellation_check=cancellation_check,
    )
    registry.view.assert_current(geometry)
    return AuthoredPlanarStationPlan(
        int(correspondence.authored_definition.face_id), tuple(receipts),
    )
