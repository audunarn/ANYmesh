"""Bind original-domain exterior stations to existing mesh node identities.

This is an assertion over already registered nodes. It neither allocates stations
nor grants permission to mesh or publish an authored-root domain.
"""

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from .errors import MeshError


@dataclass(frozen=True)
class BoundAuthoredExteriorStations:
    edge_id: int
    node_ids: tuple[int, ...]
    parameters: tuple[float, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    authored_uv: tuple[tuple[float, float], ...]
    owner_receipt: object

    @property
    def publication_qualified(self) -> bool:
        return False


def bind_authored_exterior_stations(
    geometry, correspondence, mesh, registry, edge_id, cancellation_check=None,
) -> BoundAuthoredExteriorStations:
    """Assert exact registry identity and owner-qualified original coordinates.

    ``coordinates`` retain the existing mesh XYZ, not a projection or owner
    replacement. Re-run this binding after any candidate topology transaction.
    """
    from anygeometry import (
        query_prepared_authored_boundary_stations,
        validate_prepared_authored_boundary_station_coordinates,
    )

    if registry.view.source is not geometry:
        raise MeshError("authored boundary registry belongs to another geometry owner")
    registry.view.assert_current(geometry)
    if mesh.geometry_model_id is not None and str(mesh.geometry_model_id) != str(geometry.model_id):
        raise MeshError("authored boundary mesh belongs to another geometry owner")
    if mesh.geometry_revision is not None and mesh.geometry_revision != geometry.revision:
        raise MeshError("authored boundary mesh revision is stale")
    if isinstance(edge_id, bool) or not isinstance(edge_id, Integral) or edge_id <= 0:
        raise MeshError("authored boundary needs a positive source edge ID")
    edge_id = int(edge_id)
    nodes = tuple(mesh.nodes_of_edge.get(edge_id, ()))
    entries = registry.entries(edge_id)
    if len(nodes) < 2 or len(nodes) != len(set(nodes)) or len(entries) != len(nodes):
        raise MeshError("authored boundary has incomplete or duplicate registered stations")
    owner = geometry.handle("edge", edge_id)
    by_node = {}
    for entry in entries:
        if entry.node_id is None or entry.node_id in by_node or owner not in entry.owners:
            raise MeshError("authored boundary station has no unique source-edge identity")
        by_node[entry.node_id] = entry
    if set(by_node) != set(nodes):
        raise MeshError("authored boundary node sequence differs from registered stations")
    ordered = tuple(sorted(nodes, key=lambda node: by_node[node].key.parameter))
    parameters = tuple(float(by_node[node].key.parameter) for node in ordered)
    if parameters[0] != 0.0 or parameters[-1] != 1.0:
        raise MeshError("authored boundary is missing an endpoint station")
    coordinates = []
    for node in ordered:
        try:
            point = np.asarray(mesh.nodes.get(node), dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise MeshError("authored boundary has an invalid registered node") from error
        if point.shape != (3,) or not np.isfinite(point).all():
            raise MeshError("authored boundary has an invalid registered node")
        if not np.array_equal(point, by_node[node].point):
            raise MeshError("authored boundary node XYZ changed after registration")
        coordinates.append(tuple(float(value) for value in point))
    receipt = query_prepared_authored_boundary_stations(
        geometry, correspondence, edge_id,
        tuple(Fraction.from_float(value) for value in parameters),
        cancellation_check=cancellation_check,
    )
    validate_prepared_authored_boundary_station_coordinates(
        geometry, receipt, coordinates, cancellation_check=cancellation_check,
    )
    uv = tuple(tuple(float(Fraction(*value)) for value in pair)
               for pair in receipt.authored_uv)
    if len(uv) != len(ordered) or not np.isfinite(np.asarray(uv)).all():
        raise MeshError("authored boundary owner returned invalid original coordinates")
    return BoundAuthoredExteriorStations(edge_id, ordered, parameters,
                                        tuple(coordinates), uv, receipt)
