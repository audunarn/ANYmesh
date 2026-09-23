"""Topology-owned cylindrical fixtures for end-to-end native qualification.

These are authored sector faces, not a single-face periodic representation.
No coordinate welding, test-module imports, or meshing monkeypatches are used.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any

import numpy as np

CYLINDER_CASES = ("cylinder_patch", "cylinder_sectors", "cylinder_seam_trim")


@dataclass(frozen=True)
class CylinderBenchmark:
    model: Any
    face_ids: tuple[int, ...]
    area: float
    edge_lengths: tuple[tuple[int, float], ...]
    centers: tuple[tuple[float, float, float], ...]
    name: str

    def overrides(self, target: float) -> dict[int, int]:
        # Identical immutable protected stations for the two compared routes.
        return {edge: max(1, math.ceil(length / target))
                for edge, length in self.edge_lengths}

    def refinements(self, target: float):
        from anymesher.refinement import Refinement
        return tuple(
            Refinement(size=target * .5, radius=.15, center=center,
                       growth=1.5, name=f"interior-{index}")
            for index, center in enumerate(self.centers)
        )

    def metric_spec(self, target: float):
        from anymesher import (
            FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec,
        )
        return MetricFieldSpec(
            IsotropicMetricControl(target),
            feature_controls=tuple(
                FeatureDistanceMetricControl(
                    (tuple(zone.center),), zone.size, zone.radius, zone.growth, zone.name)
                for zone in self.refinements(target)
            ),
        )

    def options(self, target: float, route: str, requested_elements: int):
        from anymesher import NativeMeshingOptions
        if route not in {"legacy", "frontal"}:
            raise ValueError("unknown benchmark route")
        if route == "legacy":
            # Spatial sizing uses the established public refinements argument.
            # It must not activate native-v2 through legacy NativeMeshingOptions.
            return NativeMeshingOptions()
        return NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            # The public refinements argument is the sole sizing authority.
            metric_field=None,
            max_insertions=max(128, requested_elements // 2),
        )

    def mesh_contract(self, mesh, target: float) -> dict[str, Any]:
        """Check actual per-face incidence and protected-node ownership.

        Interior connectivity/counts are deliberately not a cross-route oracle.
        Protected topology and coordinates are an exact cross-route oracle.
        """
        if mesh.geometry_model_id != self.model.model_id:
            raise ValueError("foreign benchmark mesh")
        if mesh.geometry_revision != self.model.revision:
            raise ValueError("stale benchmark mesh")
        if set(mesh.elements_of_face) != set(self.face_ids):
            raise ValueError("benchmark face ownership changed")
        all_elements = []
        protected = []
        for edge, divisions in sorted(self.overrides(target).items()):
            sequence = tuple(mesh.nodes_of_edge[edge])
            if len(sequence) != divisions + 1 or len(set(sequence)) != len(sequence):
                raise ValueError("protected station count/identity changed")
            owner = self.model.edges[edge]
            if (sequence[0] != mesh.node_of_vertex[owner.start]
                    or sequence[-1] != mesh.node_of_vertex[owner.end]):
                raise ValueError("protected endpoint identity changed")
            protected.append({
                "edge": edge, "nodes": sequence,
                "coordinates": np.asarray([mesh.nodes[node] for node in sequence]),
            })
        for face_id in self.face_ids:
            elements = tuple(mesh.elements_of_face[face_id])
            all_elements.extend(elements)
            counts = Counter()
            for element in elements:
                row = mesh.shells[element]
                corners = 3 if element in mesh.tris else 4
                if len(row) != corners:
                    raise ValueError("benchmark requires linear elements")
                for a, b in zip(row, (*row[1:], row[0])):
                    counts[tuple(sorted((a, b)))] += 1
            expected = set()
            face = self.model.faces[face_id]
            for loop in (face.loop, *face.holes):
                for use in loop:
                    sequence = mesh.nodes_of_edge[use.edge]
                    expected.update(tuple(sorted((a, b)))
                                    for a, b in zip(sequence, sequence[1:]))
            if {edge for edge, count in counts.items() if count == 1} != expected:
                raise ValueError("missing boundary or cracked face interior")
            if any(count != (1 if edge in expected else 2)
                   for edge, count in counts.items()):
                raise ValueError("invalid face incidence")
        if (len(all_elements) != len(set(all_elements))
                or set(all_elements) != set(mesh.shells)):
            raise ValueError("missing or duplicate element ownership")
        return {
            "recipe": self.name, "faces": self.face_ids,
            "protected": protected, "ownership_and_incidence_validated": True,
        }


def cylinder_case(name: str) -> CylinderBenchmark:
    from anygeometry import Cylinder, GeometryModel, OrientedEdge

    if name not in CYLINDER_CASES:
        raise ValueError(name)
    model = GeometryModel()
    vertices, edges, lengths = {}, {}, {}
    faces, centers = [], []
    rectangle = ((0, 0), (1, 0), (1, 2), (0, 2))
    first = ((0, 0), (1, 0), (1, 2), (0, 2),
             (0, 1.25), (.25, 1.25), (.25, .75), (0, .75))
    last = ((0, 0), (1, 0), (1, .75), (.75, .75),
            (.75, 1.25), (1, 1.25), (1, 2), (0, 2))
    count = 1 if name == "cylinder_patch" else 8
    trimmed = name == "cylinder_seam_trim"

    def vertex(angle, z):
        key = (Fraction(angle) % 8, Fraction(z))
        if key not in vertices:
            theta = float(key[0]) * math.pi / 4
            vertices[key] = model.add_point(
                math.cos(theta), math.sin(theta), float(z))
        return vertices[key]

    def edge(a, b):
        ka, kb = (a[0] % 8, a[1]), (b[0] % 8, b[1])
        circular = a[1] == b[1]
        key = (circular, *sorted((ka, kb)))
        start, end = vertex(*a), vertex(*b)
        if key not in edges:
            identifier = (
                model.add_arc(start, vertex((a[0] + b[0]) / 2, a[1]), end)
                if circular else model.add_line(start, end)
            )
            edges[key] = identifier
            lengths[identifier] = (
                float(abs(a[0] - b[0])) * math.pi / 4
                if circular else float(abs(a[1] - b[1]))
            )
        identifier = edges[key]
        return OrientedEdge(identifier, model.edges[identifier].start == start)

    with model.transaction():
        for index in range(count):
            local = first if trimmed and index == 0 else (
                last if trimmed and index == 7 else rectangle)
            points = tuple((Fraction(index) + Fraction(u), Fraction(z))
                           for u, z in local)
            loop = tuple(edge(a, b)
                         for a, b in zip(points, (*points[1:], points[0])))
            surface = Cylinder(
                origin=(0., 0., 0.), axis=(0., 0., 1.),
                radial_direction=(1., 0., 0.), radius=1., height=2.,
                start_angle=index * math.pi / 4, sweep_angle=math.pi / 4,
            )
            faces.append(model.add_face_from_loop(loop, surface=surface))
            theta = (index + .5) * math.pi / 4
            # Keep physical refinement features away from the seam-trim notch.
            centers.append((math.cos(theta), math.sin(theta), .4 if trimmed else 1.))
        part = model.add_part(name="native cylinder benchmark")
        model.add_sheet(faces, part_id=part)
    area = count * math.pi / 2 - (math.pi / 16 if trimmed else 0.)
    return CylinderBenchmark(model, tuple(faces), area,
                             tuple(sorted(lengths.items())), tuple(centers), name)
