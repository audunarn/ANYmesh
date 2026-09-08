"""Synchronize existing owner-bound midpoint stations before face refinement."""
import numpy as np

from .errors import MeshError
from ._shared_triangle_split import propagate_triangle_split


def synchronize_existing_midpoints(source, working, stage, registry, faces,
                                    eligible_edges, faces_using_edge, projection,
                                    incidence_cache, max_face_operations, checkpoint):
    face_set = set(faces)
    operations = dict.fromkeys(faces, 0)
    plans = []
    for edge in sorted(set(eligible_edges)):
        incident = tuple(sorted(set(faces_using_edge(edge))))
        if not incident or not set(incident).issubset(face_set):
            raise MeshError("midpoint preparation requires complete component ownership")
        sequence = source.nodes_of_edge[edge]
        if len(sequence) < 3 or len(sequence) % 2 != 1:
            raise MeshError("midpoint preparation requires a complete quadratic edge")
        if list(working.nodes_of_edge[edge]) != list(sequence[::2]):
            raise MeshError("midpoint preparation must precede dynamic edge splitting")
        for offset in range(0, len(sequence) - 2, 2):
            first, middle, last = sequence[offset:offset + 3]
            if np.asarray(working.nodes[middle]).tobytes() != np.asarray(source.nodes[middle]).tobytes():
                raise MeshError("midpoint preparation encountered changed owner coordinates")
            plans.append((edge, first, middle, last, incident))
            for face in incident:
                operations[face] += 1
    if any(count > max_face_operations for count in operations.values()):
        raise MeshError("midpoint preparation exceeds component topology budget")

    receipts = []
    for edge, first, middle, last, incident in plans:
        checkpoint("quadratic midpoint preparation")
        propose = stage.split_proposal(
            lambda edge_id, node_id, physical: projection(incident[0], (physical,))[0]
        )
        lower, upper = stage.station(edge, first), stage.station(edge, last)
        proposal = propose(edge, lower, upper)
        if proposal is None or proposal[2] != middle:
            raise MeshError("midpoint preparation lost its exact reserved station")
        station = proposal[0]

        def publish(node):
            if node != middle:
                raise MeshError("midpoint preparation allocated a replacement identity")
            propagate_triangle_split(working, incident, (first, last), node, cache=incidence_cache)
            sequence = working.nodes_of_edge[edge]
            position = sequence.index(first)
            if position + 1 >= len(sequence) or sequence[position + 1] != last:
                raise MeshError("midpoint preparation changed edge order")
            sequence.insert(position + 1, node)
            stage.record_split(edge, first, last, node, station)

        registry._resolve_and_publish(edge, station.numerator, station.denominator,
                                      publish, existing_node_id=middle)
        receipts.append({"edge_id": edge, "node_id": middle,
                         "station": [station.numerator, station.denominator],
                         "incident_faces": list(incident)})
    checkpoint("quadratic midpoint preparation complete")
    return {"reused_node_count": len(receipts), "new_node_count": 0,
            "face_operations": {str(face): operations[face] for face in sorted(faces)},
            "stations": receipts}
