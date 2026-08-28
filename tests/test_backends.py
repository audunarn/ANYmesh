"""Backend dispatch, and the gmsh backend.

The gmsh tests are skipped when gmsh is not installed, since it is an optional
extra.  What they check is not that gmsh works -- that is gmsh's problem -- but
that what comes back is a mesh this package's own machinery can use, and that the
two guarantees the mapped mesher makes and gmsh does not are stated rather than
quietly broken.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from anygeometry import punch_hole

from anymesher import (
    DEFAULT_BACKEND,
    EntityRef,
    GeometryModel,
    MeshError,
    available_backends,
    generate_mesh,
    resolve_backend,
    verify_mesh_quality,
)

gmsh = pytest.importorskip("gmsh", reason="the gmsh backend needs ANYmesher[gmsh]")

from anymesher.backends import gmsh as gmsh_backend


def _rectangle(length: float = 2.0, width: float = 1.0) -> GeometryModel:
    model = GeometryModel()
    model.add_face(
        model.add_polyline(
            model.add_points([(0, 0, 0), (length, 0, 0), (length, width, 0), (0, width, 0)]),
            close=True,
        )
    )
    return model


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 environment contract")
@pytest.mark.parametrize(
    "value",
    [None, "", "C:\\værktøy;D:\\工具", "C:\\long-path;" + "x" * 5000],
)
def test_native_environment_snapshot_round_trips_all_path_states(
    value: str | None,
) -> None:
    name = f"ANYMESHER_NATIVE_PATH_TEST_{os.getpid()}"
    original = gmsh_backend._snapshot_native_environment_variable(name)
    python_value = os.environ.get(name)
    requested = (False, "") if value is None else (True, value)
    try:
        gmsh_backend._restore_native_environment_variable(name, requested)
        captured = gmsh_backend._snapshot_native_environment_variable(name)
        assert captured == requested
        gmsh_backend._restore_native_environment_variable(
            name, (True, "changed-after-snapshot")
        )
        gmsh_backend._restore_native_environment_variable(name, captured)
        assert gmsh_backend._snapshot_native_environment_variable(name) == requested
        assert os.environ.get(name) == python_value
    finally:
        gmsh_backend._restore_native_environment_variable(name, original)


def test_gmsh_session_restores_immediately_and_after_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    snapshot = (True, "frozen-native-path")

    class _Option:
        @staticmethod
        def setNumber(name: str, value: int) -> None:
            assert (name, value) == ("General.Terminal", 0)
            events.append("option")

    fake = SimpleNamespace(
        initialize=lambda: events.append("initialize"),
        finalize=lambda: events.append("finalize"),
        option=_Option(),
    )
    monkeypatch.setattr(gmsh_backend, "gmsh", fake)
    monkeypatch.setattr(
        gmsh_backend,
        "_snapshot_native_environment_variable",
        lambda name: snapshot if name == "PATH" else None,
    )
    monkeypatch.setattr(
        gmsh_backend,
        "_restore_native_environment_variable",
        lambda name, value: events.append(f"restore:{name}:{value!r}"),
    )

    with gmsh_backend._gmsh_session():
        events.append("body")

    assert events == [
        "initialize",
        f"restore:PATH:{snapshot!r}",
        "option",
        "body",
        "finalize",
        f"restore:PATH:{snapshot!r}",
    ]


def test_gmsh_session_restores_after_initialize_and_finalize_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    snapshot = (True, "frozen-native-path")

    def restore(name: str, value: object) -> None:
        events.append(f"restore:{name}:{value!r}")

    failing_initialize = SimpleNamespace(
        initialize=lambda: (_ for _ in ()).throw(RuntimeError("initialize failed")),
        finalize=lambda: events.append("unexpected-finalize"),
        option=SimpleNamespace(setNumber=lambda *_: None),
    )
    monkeypatch.setattr(gmsh_backend, "gmsh", failing_initialize)
    monkeypatch.setattr(
        gmsh_backend,
        "_snapshot_native_environment_variable",
        lambda name: snapshot if name == "PATH" else None,
    )
    monkeypatch.setattr(
        gmsh_backend, "_restore_native_environment_variable", restore
    )

    with pytest.raises(RuntimeError, match="initialize failed"):
        with gmsh_backend._gmsh_session():
            pytest.fail("initialization failure must not enter the body")
    assert events == [
        f"restore:PATH:{snapshot!r}",
        f"restore:PATH:{snapshot!r}",
    ]

    events.clear()

    def finalize() -> None:
        events.append("finalize")
        raise RuntimeError("finalize failed")

    failing_finalize = SimpleNamespace(
        initialize=lambda: events.append("initialize"),
        finalize=finalize,
        option=SimpleNamespace(setNumber=lambda *_: events.append("option")),
    )
    monkeypatch.setattr(gmsh_backend, "gmsh", failing_finalize)
    with pytest.raises(RuntimeError, match="finalize failed"):
        with gmsh_backend._gmsh_session():
            events.append("body")
    assert events == [
        "initialize",
        f"restore:PATH:{snapshot!r}",
        "option",
        "body",
        "finalize",
        f"restore:PATH:{snapshot!r}",
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="Gmsh Win32 PATH regression")
def test_real_gmsh_mesh_preserves_native_path_and_relative_git_launch() -> None:
    native_before = gmsh_backend._snapshot_native_environment_variable("PATH")
    python_before = os.environ.get("PATH")
    git = shutil.which("git")
    assert git is not None and os.path.isfile(git)

    mesh = generate_mesh(_rectangle(), backend="gmsh", target_size=0.25)

    assert mesh.shells
    assert gmsh_backend._snapshot_native_environment_variable("PATH") == native_before
    assert os.environ.get("PATH") == python_before
    completed = subprocess.run(
        ["git", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("git version ")


def test_the_default_backend_is_the_native_first_auto_dispatcher() -> None:
    assert DEFAULT_BACKEND == "auto"
    assert available_backends() == ("auto", "gmsh", "mapped", "native")

    from anymesher.hybrid import generate_hybrid_mesh
    from anymesher.mapped import generate_mesh as mapped_generate

    assert resolve_backend("auto") is generate_hybrid_mesh
    assert resolve_backend("AUTO") is generate_hybrid_mesh
    assert resolve_backend("mapped") is mapped_generate
    assert resolve_backend("MAPPED") is mapped_generate


def test_dispatching_with_the_default_matches_calling_the_mesher_directly() -> None:
    from anymesher.mapped import generate_mesh as mapped_generate

    dispatched = generate_mesh(_rectangle(), target_size=0.25)
    direct = mapped_generate(_rectangle(), target_size=0.25)

    assert dispatched.quads == direct.quads
    assert all(np.array_equal(dispatched.nodes[k], direct.nodes[k]) for k in direct.nodes)


def test_an_unknown_backend_names_the_ones_that_exist() -> None:
    with pytest.raises(MeshError, match="unknown mesh backend 'quadratic'"):
        generate_mesh(_rectangle(), backend="quadratic", target_size=0.25)
    with pytest.raises(MeshError, match="auto, gmsh, mapped, native"):
        resolve_backend("nope")


@pytest.mark.parametrize("order,nodes_per_quad", [("linear", 4), ("quadratic", 8)])
def test_gmsh_returns_a_usable_mesh(order: str, nodes_per_quad: int) -> None:
    mesh = generate_mesh(_rectangle(), backend="gmsh", target_size=0.25, order=order)

    assert mesh.order == order
    assert mesh.quads
    assert all(len(nodes) == nodes_per_quad for nodes in mesh.quads.values())
    # Every element node exists, or nothing downstream can look one up.
    assert all(node in mesh.nodes for nodes in mesh.shells.values() for node in nodes)
    # Serendipity, not Lagrange: no centre node on a quadratic quad.
    assert nodes_per_quad != 9

    quality = verify_mesh_quality(mesh)
    assert quality.num_shell_elements == len(mesh.quads) + len(mesh.tris)
    assert np.isfinite(quality.max_aspect_ratio)


def test_gmsh_fills_the_association_so_results_stay_addressable() -> None:
    model = _rectangle()
    mesh = generate_mesh(model, backend="gmsh", target_size=0.25)

    assert mesh.elements_of_face[1]
    assert mesh.nodes_on(EntityRef("face", 1))
    for edge_id in model.edges:
        nodes = mesh.nodes_on(EntityRef("edge", edge_id))
        assert len(nodes) >= 2
        # Ordered along the edge: consecutive nodes must be monotonic in arc
        # length, or a load distributed along a line lands out of order.
        positions = np.array([mesh.nodes[node] for node in nodes])
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert np.all(steps > 0.0)
        span = float(np.linalg.norm(positions[-1] - positions[0]))
        assert float(np.sum(steps)) == pytest.approx(span, rel=1.0e-6)
    for vertex_id in model.vertices:
        assert vertex_id in mesh.node_of_vertex


def test_gmsh_has_no_structured_grid_and_says_so_by_leaving_it_empty() -> None:
    mesh = generate_mesh(_rectangle(), backend="gmsh", target_size=0.3)

    # The mapped mesher's (i, j) index does not exist for an unstructured mesh.
    # It is left empty rather than filled with something plausible.
    assert mesh.grid_of_face == {}
    # nodes_on still works, through the element groups instead of the grid.
    assert mesh.nodes_on(EntityRef("face", 1))


def test_gmsh_keeps_triangles_rather_than_dropping_them() -> None:
    mesh = generate_mesh(_rectangle(), backend="gmsh", target_size=0.3, recombine=False)

    assert mesh.tris
    assert not mesh.quads
    assert mesh.num_elements == len(mesh.tris)
    assert all(len(nodes) == 3 for nodes in mesh.tris.values())
    assert all(node in mesh.nodes for nodes in mesh.tris.values() for node in nodes)


def test_gmsh_consumes_a_neutral_triangular_face_directly() -> None:
    model = GeometryModel()
    vertices = model.add_points([(0, 0, 0), (1, 0, 0), (0.25, 0.8, 0)])
    face = model.add_face(model.add_polyline(vertices, close=True))

    mesh = generate_mesh(
        model, backend="gmsh", target_size=0.2, face_ids=[face], recombine=False
    )

    assert mesh.tris
    assert mesh.elements_on(EntityRef("face", face)) == sorted(mesh.tris)


def test_gmsh_respects_neutral_face_holes() -> None:
    model = _rectangle(2.0, 2.0)
    _face, hole_edges = punch_hole(model, 1, (1.0, 1.0, 0.0), 0.3)

    mesh = generate_mesh(
        model, backend="gmsh", target_size=0.15, recombine=False
    )

    ring = np.vstack(
        [
            [mesh.nodes[node] for node in mesh.nodes_on(EntityRef("edge", edge))]
            for edge in hole_edges
        ]
    )
    assert np.linalg.norm(ring[:, :2] - np.array([1.0, 1.0]), axis=1) == pytest.approx(
        0.3, rel=1.0e-6
    )
    connected_nodes = sorted({node for element in mesh.shells.values() for node in element})
    connected = np.asarray([mesh.nodes[node] for node in connected_nodes])
    assert np.linalg.norm(connected[:, :2] - np.array([1.0, 1.0]), axis=1).min() == pytest.approx(
        0.3, rel=1.0e-6
    )


def test_gmsh_puts_arc_nodes_exactly_on_the_arc() -> None:
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    via = model.add_point(1.0, 1.2, 0.0)
    edges = [
        model.add_line(points[0], points[1]),
        model.add_line(points[1], points[2]),
        model.add_arc(points[2], via, points[3]),
        model.add_line(points[3], points[0]),
    ]
    model.add_face(edges)

    mesh = generate_mesh(model, backend="gmsh", target_size=0.25)
    frame = model.arc_frame(edges[2])
    radii = np.linalg.norm(
        np.array([mesh.nodes[node] for node in mesh.nodes_on(EntityRef("edge", edges[2]))])
        - frame.center,
        axis=1,
    )
    assert radii == pytest.approx(frame.radius)


def test_gmsh_rebuilds_owner_bezier_splines_without_polyline_approximation() -> None:
    model = GeometryModel()
    start, control_1, control_2, end, upper_right, upper_left = model.add_points(
        [
            (0.0, 0.0, 0.0),
            (0.5, -0.3, 0.0),
            (1.5, -0.3, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    spline = model.add_spline(start, (control_1, control_2), end)
    boundary = [
        spline,
        model.add_line(end, upper_right),
        model.add_line(upper_right, upper_left),
        model.add_line(upper_left, start),
    ]
    model.add_face(boundary)

    mesh = generate_mesh(model, backend="gmsh", target_size=0.15)
    spline_nodes = mesh.nodes_on(EntityRef("edge", spline))
    samples = model.sample_edge(spline, np.linspace(0.0, 1.0, 20001))
    distances = [
        float(np.linalg.norm(samples - mesh.nodes[node], axis=1).min())
        for node in spline_nodes
    ]

    assert len(spline_nodes) >= 3
    assert max(distances) < 1.0e-4
    connected = {node for element in mesh.shells.values() for node in element}
    assert set(mesh.nodes) == connected
    assert control_1 not in mesh.node_of_vertex
    assert control_2 not in mesh.node_of_vertex


def test_gmsh_refuses_a_non_planar_face_and_points_at_the_mapped_mesher() -> None:
    model = GeometryModel()
    radius = 2.0
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    model.extrude([model.add_arc(start, via, end)], (0.0, 0.0, 3.0))

    with pytest.raises(MeshError, match="not planar"):
        generate_mesh(model, backend="gmsh", target_size=0.4)
    with pytest.raises(MeshError, match="backend='mapped'"):
        generate_mesh(model, backend="gmsh", target_size=0.4)

    # The mapped mesher meshes it exactly, which is why refusing is reasonable.
    assert generate_mesh(model, target_size=0.4).quads


def test_gmsh_validates_its_own_inputs() -> None:
    model = _rectangle()

    with pytest.raises(MeshError, match="target_size must be positive"):
        generate_mesh(model, backend="gmsh", target_size=0.0)
    with pytest.raises(MeshError, match="unknown element order"):
        generate_mesh(model, backend="gmsh", target_size=0.3, order="cubic")
    with pytest.raises(MeshError, match="no face 99"):
        generate_mesh(model, backend="gmsh", target_size=0.3, face_ids=[99])
    with pytest.raises(MeshError, match="no edge 99"):
        generate_mesh(model, backend="gmsh", target_size=0.3, beam_edges=[99])


def test_gmsh_is_finalized_even_when_meshing_fails() -> None:
    model = GeometryModel()
    radius = 2.0
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    model.extrude([model.add_arc(start, via, end)], (0.0, 0.0, 3.0))

    with pytest.raises(MeshError):
        generate_mesh(model, backend="gmsh", target_size=0.4)

    # gmsh is process-global state.  A backend that left it initialized would
    # break the next unrelated call, so it must report clean state and allow a
    # second backend-owned session.
    assert not gmsh.isInitialized()
    assert generate_mesh(_rectangle(), backend="gmsh", target_size=0.3).quads


def test_gmsh_meshes_beam_edges_along_their_own_direction() -> None:
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)

    mesh = generate_mesh(model, backend="gmsh", target_size=0.25, beam_edges=[edges[0]])
    assert mesh.beams
    assert mesh.elements_of_edge[edges[0]]
    for connectivity in mesh.beams.values():
        assert len(connectivity) == 2
        assert all(node in mesh.nodes for node in connectivity)
