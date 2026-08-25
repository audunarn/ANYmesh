"""Cross-package contract for the ANYworkspaceAI Phase 1B package set."""

from __future__ import annotations

import anygeometry as ag
import anymesher as am
from anygeometry.generators import plate


def test_anygeometry_0_3_hybrid_mesh_preserves_source_identity() -> None:
    geometry = plate(2.0, 1.0, semantic_group="qualification_plate")
    source_id = geometry.model_id
    source_revision = geometry.revision
    source_document = ag.to_dict(geometry)

    result = am.generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        strategy="auto",
        structural_preparation=False,
        certification_mode="interactive",
    )
    mesh = result.mesh

    assert ag.__version__.startswith("0.3.")
    assert str(mesh.geometry_model_id) == str(source_id)
    assert mesh.geometry_revision == source_revision
    assert geometry.revision == source_revision
    assert ag.to_dict(geometry) == source_document
    assert mesh.num_nodes > 0
    assert mesh.shells

    face_id = next(iter(geometry.faces))
    assert mesh.elements_on(geometry.handle("face", face_id))

    quality = am.verify_mesh_quality(mesh)
    assert quality.num_shell_elements == len(mesh.quads) + len(mesh.tris)
