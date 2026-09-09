"""Installed-wheel smoke for the three hybrid routes and structured planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anygeometry as ag
import anymesher as am


def _quad() -> tuple[ag.GeometryModel, int]:
    geometry = ag.GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    )
    return geometry, face


def _triangle() -> tuple[ag.GeometryModel, int]:
    geometry = ag.GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (0.5, 1.5, 0)))
    )
    return geometry, face


def _cylindrical_frontal(require_native: bool) -> dict:
    # Only the geometry recipe comes from the checkout. All meshing and owner
    # implementation imports remain the installed packages imported above.
    fixture_path = str(Path(__file__).resolve().parents[1] / "benchmarks")
    sys.path.insert(0, fixture_path)
    try:
        from native_v2_cylinder_cases import cylinder_case
    finally:
        sys.path.remove(fixture_path)
    from anymesher.hybrid import _neutral_shell_core
    from anymesher.quality_v2 import assert_valid_mesh, evaluate_quality

    fixture = cylinder_case("cylinder_sectors")
    before = json.dumps(ag.to_dict(fixture.model), sort_keys=True)
    target = 0.25
    result = am.generate_hybrid_mesh_result(
        fixture.model, face_ids=fixture.face_ids, target_size=target,
        strategy="native", overrides=fixture.overrides(target),
        refinements=fixture.refinements(target), order="linear", recombine=True,
        native_backend="native" if require_native else "python",
        native_options=fixture.options(target, "frontal", 256),
    )
    assert set(result.strategy_by_face.values()) == {"native"}
    fixture.mesh_contract(result.mesh, target)
    core = _neutral_shell_core(result.mesh)
    assert_valid_mesh(core)
    quality = evaluate_quality(core)
    assert quality.minimum_scaled_jacobian > 0
    assert quality.maximum_aspect_ratio <= 5
    rows = result.triangulation_backend_by_face.values()
    insertions = sum(row.get("native_v2", {}).get("published_insertions", 0)
                     for row in rows)
    assert insertions > 0, "Installed cylinder smoke must exercise refinement"
    if require_native:
        assert all(row["actual_backend"] == "anymesher-cpp17" for row in rows)
    assert json.dumps(ag.to_dict(fixture.model), sort_keys=True) == before
    return {"case": fixture.name, "faces": len(fixture.face_ids),
            "published_insertions": insertions, "nodes": len(result.mesh.nodes),
            "elements": len(result.mesh.shells), "protected_contract": "passed",
            "minimum_scaled_jacobian": quality.minimum_scaled_jacobian}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--require-native", action="store_true")
    options = parser.parse_args()

    origin = Path(am.__file__).resolve()
    source_tree = Path(__file__).resolve().parents[1] / "src"
    assert source_tree not in origin.parents, (origin, source_tree)
    assert am.__version__ == options.expect_version

    automatic_model, automatic_face = _quad()
    automatic = am.generate_hybrid_mesh_result(
        automatic_model,
        target_size=0.25,
        strategy="auto",
    )
    assert automatic.strategy_by_face == {automatic_face: "mapped"}
    assert automatic.mesh.quads
    assert automatic.structural_preparation is not None

    mapped_model, mapped_face = _quad()
    mapped = am.generate_hybrid_mesh_result(
        mapped_model,
        target_size=0.25,
        strategy="mapped",
    )
    assert mapped.strategy_by_face == {mapped_face: "mapped"}
    assert mapped.mesh.quads

    native_model, native_face = _quad()
    native = am.generate_hybrid_mesh_result(
        native_model,
        target_size=0.25,
        strategy="native",
        native_backend="native" if options.require_native else "python",
    )
    assert native.strategy_by_face == {native_face: "native"}
    assert native.mesh.tris
    if options.require_native:
        assert native.triangulation_backend_by_face[native_face][
            "actual_backend"
        ] == "anymesher-cpp17"

    structured_model, structured_face = _triangle()
    structured = am.generate_hybrid_mesh_result(
        structured_model,
        target_size=0.25,
        strategy="auto",
        structured_options={},
    )
    assert structured.strategy_by_face == {structured_face: "mapped"}
    assert structured.structured_layout is not None
    assert structured.structured_layout.status == "applied"
    assert structured.mesh.quads

    cylindrical = _cylindrical_frontal(options.require_native)

    print(
        json.dumps(
            {
                "origin": str(origin),
                "version": am.__version__,
                "automatic": automatic.strategy_by_face[automatic_face],
                "mapped": mapped.strategy_by_face[mapped_face],
                "native": native.triangulation_backend_by_face[native_face][
                    "actual_backend"
                ],
                "structured": structured.structured_layout.status,
                "cylindrical_frontal": cylindrical,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
