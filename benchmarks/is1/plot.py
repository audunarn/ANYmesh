"""Render compact, reproducible views from an immutable IS1 attempt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from anymesher.serialize import mesh_from_dict
from benchmarks.is1.fixtures import CASES, SIZES


def render(attempt: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        h = SIZES[case][1]
        folder = attempt / f"{case}-h{h}-quadratic-uniform-adaptive"
        mesh = mesh_from_dict(json.loads((folder / "mesh.json").read_text()))
        metrics = json.loads((folder / "metrics.json").read_text())
        planar = case.startswith("planar_")
        figure = plt.figure(figsize=(7.5, 6.0), constrained_layout=True)
        axes = figure.add_subplot(111 if planar else 111, projection=None if planar else "3d")
        segments, colors = [], []
        for family, elements in (("Q8", mesh.quads), ("T6", mesh.tris)):
            corners = 4 if family == "Q8" else 3
            for element, body in elements.items():
                points = np.asarray([mesh.nodes[node] for node in body[:corners]], dtype=float)
                for first, second in zip(points, np.roll(points, -1, axis=0)):
                    segments.append(np.stack((first, second))[:, :2 if planar else 3])
                    colors.append("#1f77b4" if family == "Q8" else "#e07020")
        if planar:
            axes.add_collection(LineCollection(segments, colors=colors, linewidths=.85))
            axes.autoscale()
            axes.set_aspect("equal", adjustable="box")
            axes.set_xlabel("x (m)")
            axes.set_ylabel("y (m)")
        else:
            axes.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=.75))
            points = np.asarray(list(mesh.nodes.values()), dtype=float)
            for i, axis in enumerate((axes.set_xlim, axes.set_ylim, axes.set_zlim)):
                axis(float(points[:, i].min()), float(points[:, i].max()))
            span = np.ptp(points, axis=0)
            axes.set_box_aspect(np.maximum(span, .1))
            axes.set_xlabel("x (m)")
            axes.set_ylabel("y (m)")
            axes.set_zlabel("z (m)")
            axes.view_init(elev=24, azim=-58)
        axes.set_title(
            f"IS1 {case.replace('_', ' ')} · h={h:g} m\n"
            f"{len(mesh.quads)} Q8 / {len(mesh.tris)} T6 · "
            f"min sampled Jacobian {metrics['min_normalized_jacobian']:.3f}"
        )
        figure.savefig(output / f"{case}-fine-quadratic.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    render(arguments.attempt, arguments.output)
