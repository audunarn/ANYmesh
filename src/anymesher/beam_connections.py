"""Mesh-level connection of independently modelled beams and shells."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np

from .coupling import shape_functions_4node, shape_functions_8node
from .mesh import Coupling, Mesh

__all__ = ["connect_beam_mesh", "connect_shell_boundaries"]


def _scale_and_tolerance(mesh: Mesh) -> tuple[float, float]:
    if not mesh.nodes:
        return 1.0, 1.0e-9
    coordinates = np.asarray(list(mesh.nodes.values()), dtype=float)
    scale = max(float(np.ptp(coordinates, axis=0).max()), 1.0)
    return scale, 1.0e-8 * scale


def _cell(point: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    return tuple(int(np.floor(float(value) / tolerance)) for value in point)  # type: ignore[return-value]


def _replace_zero_offset_coincident_nodes(mesh: Mesh, tolerance: float) -> int:
    """Merge only beam nodes, preferring an existing shell node."""

    shell_nodes = {
        int(node)
        for connectivity in mesh.shells.values()
        for node in connectivity
    }
    coupled = {int(item.beam_node) for item in mesh.couplings.values()}
    beam_nodes = {
        int(node)
        for connectivity in mesh.beams.values()
        for node in connectivity
        if int(node) not in coupled
    }
    candidates = sorted(shell_nodes | beam_nodes, key=lambda node: (node not in shell_nodes, node))
    buckets: dict[tuple[int, int, int], list[int]] = {}
    replacement: dict[int, int] = {}
    for node in candidates:
        point = mesh.nodes[node]
        cell = _cell(point, tolerance)
        match = None
        for delta in product((-1, 0, 1), repeat=3):
            nearby = tuple(cell[index] + delta[index] for index in range(3))
            for other in buckets.get(nearby, ()):
                if float(np.linalg.norm(mesh.nodes[other] - point)) <= tolerance:
                    match = other
                    break
            if match is not None:
                break
        if match is None:
            buckets.setdefault(cell, []).append(node)
        elif node in beam_nodes:
            replacement[node] = match

    if not replacement:
        return 0
    for element, connectivity in list(mesh.beams.items()):
        mesh.beams[element] = tuple(replacement.get(node, node) for node in connectivity)
    for edge, sequence in list(mesh.nodes_of_edge.items()):
        mesh.nodes_of_edge[edge] = [replacement.get(node, node) for node in sequence]
    for vertex, node in list(mesh.node_of_vertex.items()):
        mesh.node_of_vertex[vertex] = replacement.get(node, node)
    referenced = {
        int(node)
        for connectivity in list(mesh.shells.values()) + list(mesh.beams.values())
        for node in connectivity
    }
    referenced.update(
        int(node)
        for coupling in mesh.couplings.values()
        for node in (coupling.beam_node, *coupling.plate_nodes)
    )
    for old in replacement:
        if old not in referenced:
            mesh.nodes.pop(old, None)
    return len(replacement)


def _weights(node_count: int, xi: float, eta: float) -> np.ndarray:
    if node_count == 4:
        return shape_functions_4node(xi, eta)
    if node_count == 8:
        return shape_functions_8node(xi, eta)
    raise ValueError("only Q4/Q8 interpolation is supported")


def _locate_on_quad(
    point: np.ndarray,
    node_ids: tuple[int, ...],
    mesh: Mesh,
    tolerance: float,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray] | None:
    coordinates = np.asarray([mesh.nodes[node] for node in node_ids], dtype=float)
    corners = coordinates[:4]
    lower, upper = corners.min(axis=0) - tolerance, corners.max(axis=0) + tolerance
    if np.any(point < lower) or np.any(point > upper):
        return None
    xi = eta = 0.0
    for _ in range(15):
        weights = _weights(len(node_ids), xi, eta)
        current = weights @ coordinates
        epsilon = 1.0e-6
        dx = (
            _weights(len(node_ids), xi + epsilon, eta)
            - _weights(len(node_ids), xi - epsilon, eta)
        ) @ coordinates / (2.0 * epsilon)
        de = (
            _weights(len(node_ids), xi, eta + epsilon)
            - _weights(len(node_ids), xi, eta - epsilon)
        ) @ coordinates / (2.0 * epsilon)
        delta, *_ = np.linalg.lstsq(np.column_stack((dx, de)), point - current, rcond=None)
        xi += float(delta[0])
        eta += float(delta[1])
        if float(np.linalg.norm(delta)) <= 1.0e-11:
            break
    weights = _weights(len(node_ids), xi, eta)
    interpolated = weights @ coordinates
    if (
        xi < -1.0 - 1.0e-8
        or xi > 1.0 + 1.0e-8
        or eta < -1.0 - 1.0e-8
        or eta > 1.0 + 1.0e-8
        or float(np.linalg.norm(interpolated - point)) > tolerance
    ):
        return None
    return node_ids, weights, interpolated


def _shell_connection(
    point: np.ndarray,
    mesh: Mesh,
    tolerance: float,
    *,
    exclude_elements: Iterable[int] = (),
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray] | None:
    excluded = set(int(item) for item in exclude_elements)
    for element in sorted(mesh.quads):
        if element in excluded:
            continue
        located = _locate_on_quad(point, mesh.quads[element], mesh, tolerance)
        if located is not None:
            return located
    return None


def connect_beam_mesh(mesh: Mesh) -> int:
    """Connect coincident beam nodes and embed beam nodes on shell elements."""

    _scale, tolerance = _scale_and_tolerance(mesh)
    merged = _replace_zero_offset_coincident_nodes(mesh, tolerance)
    shell_nodes = {
        int(node)
        for connectivity in mesh.shells.values()
        for node in connectivity
    }
    already = {int(item.beam_node) for item in mesh.couplings.values()}
    beam_nodes = sorted(
        {
            int(node)
            for connectivity in mesh.beams.values()
            for node in connectivity
        }
    )
    next_element = max(
        (0, *mesh.quads, *mesh.tris, *mesh.beams, *mesh.couplings)
    )
    embedded = 0
    for beam_node in beam_nodes:
        if beam_node in shell_nodes or beam_node in already:
            continue
        located = _shell_connection(mesh.nodes[beam_node], mesh, tolerance)
        if located is None:
            continue
        plate_nodes, weights, interpolated = located
        next_element += 1
        mesh.couplings[next_element] = Coupling(
            beam_node=beam_node,
            plate_nodes=plate_nodes,
            weights=tuple(float(item) for item in weights),
            eccentricity=tuple(float(item) for item in mesh.nodes[beam_node] - interpolated),
        )
        embedded += 1
    return merged + embedded


def connect_shell_boundaries(mesh: Mesh, geometry) -> int:
    """Tie a shell boundary that terminates inside another shell element.

    Conformal intersections already share node IDs. This path is for a welded
    T-junction, such as the foot of a plate extruded from a sketch inside a
    support plate, where fragmenting the support into mapped regions is neither
    necessary nor always possible.
    """

    _scale, tolerance = _scale_and_tolerance(mesh)
    already = {int(item.beam_node) for item in mesh.couplings.values()}
    next_element = max((0, *mesh.quads, *mesh.tris, *mesh.beams, *mesh.couplings))
    face_of_element: dict[int, int] = {}
    for owner, elements in sorted(mesh.elements_of_face.items()):
        for element in elements:
            face_of_element.setdefault(int(element), int(owner))

    def on_face_boundary(face_id: int, point: np.ndarray) -> bool:
        face = geometry.faces.get(int(face_id))
        if face is None:
            return False
        return any(
            geometry.closest_edge_point(item.edge, point)[2] <= tolerance
            for item in face.loop
        )

    connected = 0
    for face_id in sorted(mesh.elements_of_face):
        face = geometry.faces.get(int(face_id))
        if face is None:
            continue
        own_elements = set(mesh.elements_of_face.get(face_id, ()))
        boundary_nodes = sorted(
            {
                int(node)
                for oriented in face.loop
                for node in mesh.nodes_of_edge.get(oriented.edge, ())
            }
        )
        for node in boundary_nodes:
            if node in already:
                continue
            point = mesh.nodes[node]
            located = None
            for element in sorted(mesh.quads):
                if element in own_elements or node in mesh.quads[element]:
                    continue
                candidate = _locate_on_quad(
                    point, mesh.quads[element], mesh, tolerance
                )
                if candidate is not None:
                    candidate_face = face_of_element.get(int(element))
                    if (
                        candidate_face is not None
                        and on_face_boundary(candidate_face, point)
                        and int(face_id) < candidate_face
                    ):
                        # Both plates own this intersection as a boundary.  A
                        # reciprocal interpolation pair would create an MPC
                        # cycle, so the higher stable face ID is the slave and
                        # the lower ID is the master.  A true boundary-to-
                        # interior T-junction is deliberately unaffected.
                        continue
                    located = candidate
                    break
            if located is None:
                continue
            plate_nodes, weights, interpolated = located
            next_element += 1
            mesh.couplings[next_element] = Coupling(
                beam_node=node,
                plate_nodes=plate_nodes,
                weights=tuple(float(item) for item in weights),
                eccentricity=tuple(float(item) for item in point - interpolated),
            )
            already.add(node)
            connected += 1
    return connected
