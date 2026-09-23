"""Work-bearing public quadratic cylinders without binding/staging injection."""
import json

import numpy as np
import pytest

from anymesher import (
    FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec,
    NativeMeshingOptions, hybrid,
)
from anymesher.quality_v2 import assert_valid_mesh
from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state


@pytest.mark.parametrize("recombine", (False, True), ids=("T6", "T6-Q8"))
@pytest.mark.parametrize("hole", (False, True), ids=("closed", "seam-trim"))
def test_public_quadratic_cylindrical_refinement(hole, recombine, tmp_path):
    model, _ = _sector_model(hole=hole)
    before = _persistent_state(model)
    centers = tuple((float(np.cos((i + .5) * np.pi / 4)),
                     float(np.sin((i + .5) * np.pi / 4)), 1.) for i in range(8))
    options = NativeMeshingOptions(
        point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
        metric_field=MetricFieldSpec(IsotropicMetricControl(.4), feature_controls=tuple(
            FeatureDistanceMetricControl((point,), .12, .35, 1.5, f"interior-{i}")
            for i, point in enumerate(centers))),
        max_insertions=128, max_topology_operations=20000,
    )
    result = hybrid.generate_hybrid_mesh_result(
        model, target_size=.4, strategy="native", native_backend="python",
        order="quadratic", recombine=recombine, native_options=options,
    )
    mesh = result.mesh
    assert _persistent_state(model) == before
    assert set(result.strategy_by_face.values()) == {"native"}
    assert mesh.tris and bool(mesh.quads) is recombine
    assert all(len(nodes) == 6 for nodes in mesh.tris.values())
    assert all(len(nodes) == 8 for nodes in mesh.quads.values())
    assert_valid_mesh(hybrid._neutral_shell_core(mesh))
    for face, diagnostics in result.triangulation_backend_by_face.items():
        component = diagnostics["cylindrical_component"]
        assert component["accepted"] and component["published_insertions"] > 0
        assert component["quality"]["accepted"]
        assert diagnostics["cylindrical_refinement"]["final_policy_quality"]["poor_element_ids"] == []
    for edge, sequence in mesh.nodes_of_edge.items():
        assert len(sequence) % 2 == 1 and len(sequence) == len(set(sequence))
        for face in model.faces_using_edge(edge):
            midsides = {}
            for element in mesh.elements_of_face[face]:
                nodes = mesh.shells[element]
                count = 3 if element in mesh.tris else 4
                for i in range(count):
                    pair = tuple(sorted((nodes[i], nodes[(i + 1) % count])))
                    assert pair not in midsides or midsides[pair] == nodes[count + i]
                    midsides[pair] = nodes[count + i]
            for a, middle, b in zip(sequence[::2], sequence[1::2], sequence[2::2]):
                assert midsides[tuple(sorted((a, b)))] == middle
    (tmp_path / "public_diagnostics.json").write_text(
        json.dumps(result.triangulation_backend_by_face, sort_keys=True, default=str), encoding="utf-8")
