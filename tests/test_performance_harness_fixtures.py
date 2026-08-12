"""Light regression checks for benchmark geometry fixtures."""

from __future__ import annotations

import runpy
from pathlib import Path

from anygeometry import EntityRef
from anymesher.hybrid import generate_hybrid_mesh


def test_native_pentagon_fixture_preserves_registered_boundary_order() -> None:
    harness = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "native_hybrid_performance.py")
    )
    geometry = harness["_pentagon"]()

    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.2,
        strategy="native",
    )

    assert mesh.shells
    face = geometry.faces[min(geometry.faces)]
    for oriented in face.loop:
        nodes = mesh.nodes_on(EntityRef("edge", oriented.edge))
        assert len(nodes) >= 2
        assert all(node in mesh.nodes for node in nodes)
