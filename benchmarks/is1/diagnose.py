"""Compact worst-cell diagnostic for one IS1 source fixture."""
from __future__ import annotations

import argparse
import json
import numpy as np

from benchmarks.is1.fixtures import build
from benchmarks.sg1.measure import samples
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.high_order import evaluate_mapping
from anymesher.quad.options import QuadMeshingOptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--h", type=float, default=.6)
    args = parser.parse_args()
    geometry, faces, _, overrides = build(args.case, args.h)
    mesh = generate_hybrid_mesh_result(
        geometry, target_size=args.h, face_ids=faces,
        order="quadratic", strategy="native", native_backend="python",
        quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
        layout_policy="adaptive", overrides=overrides,
    ).mesh
    records = []
    for face in faces:
        owner = geometry.faces[face].surface
        if owner is None or not hasattr(owner, "radius"):
            continue
        for eid in mesh.elements_of_face[face]:
            body = mesh.quads.get(eid, mesh.tris.get(eid))
            xyz = np.asarray([mesh.nodes[node] for node in body])
            family = ("Q" if eid in mesh.quads else "T") + str(len(body))
            result = evaluate_mapping(xyz, family, samples(eid in mesh.tris))
            worst = 0.
            worst_point = None
            for point, jac in zip(result.points, result.jacobian_vector):
                uv = owner.local_uv(tuple(point))
                radial = owner.evaluate(*uv) - owner.origin - uv[1] * owner.height * owner.axis
                radial /= np.linalg.norm(radial)
                dot = float(jac @ radial / np.linalg.norm(jac))
                error = float(np.degrees(np.arccos(np.clip(dot, -1., 1.))))
                if error > worst:
                    worst, worst_point = error, tuple(map(float, point))
            records.append({"face": face, "eid": eid, "family": family,
                            "normal_error": worst, "point": worst_point,
                            "corners": [tuple(map(float, mesh.nodes[node])) for node in body[:4 if eid in mesh.quads else 3]],
                            "node_ids": tuple(body)})
    print(json.dumps({"worst": sorted(records, key=lambda row: -row["normal_error"])[:5],
                      "repair": mesh.hybrid_diagnostics.get("quad_quality_repair"),
                      "front": mesh.hybrid_diagnostics.get("front")}, default=str)[:12000])


if __name__ == "__main__":
    main()
