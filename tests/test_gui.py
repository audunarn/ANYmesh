"""Mesher window smoke tests.

These drive the real widgets: set the entry variables, read back the status, and
check the mesh the form produced.  Skipped when no display is available, which is
the case on Linux CI runners.

One module-scoped root is used throughout; creating and destroying Tk roots per
test is unreliable on Windows.
"""

from __future__ import annotations

import tkinter as tk

import pytest

from anymesher import load_mesh, verify_mesh_quality

pytestmark = pytest.mark.gui

pytest.importorskip("tkinter.ttk", reason="the mesher window needs a tkinter build")


@pytest.fixture(scope="module")
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for tkinter")
    window.geometry("980x620+40+40")
    window.update()
    yield window
    window.destroy()


@pytest.fixture
def mesher(root):
    from anymesher.gui import MesherWindow

    frame = MesherWindow(root)
    frame.pack(fill="both", expand=True)
    root.update()
    yield frame
    frame.destroy()
    root.update()


def test_the_window_opens_with_a_meshed_panel(mesher, root) -> None:
    root.update()

    mesh = mesher.mesh
    assert mesh is not None
    assert mesh.quads and mesh.beams and mesh.couplings
    assert "nodes" in mesher.status_text
    assert "couplings" in mesher._details.cget("text")
    assert "section" in mesher._details.cget("text")


def test_dimensions_are_entered_in_mm_and_stored_in_metres(mesher, root) -> None:
    mesher._field_vars["length"].set("6000")
    mesher._field_vars["width"].set("2000")
    root.update()

    mesh = mesher.mesh
    assert mesh is not None
    positions = mesh.node_positions()
    # Entered as 6000 mm by 2000 mm, meshed as 6 m by 2 m.
    assert float(positions[:, 0].max()) == pytest.approx(6.0)
    assert float(positions[:, 1].max()) == pytest.approx(2.0)


def test_changing_divisions_re_meshes(mesher, root) -> None:
    mesher._divisions_x.set("4")
    mesher._divisions_y.set("3")
    root.update()
    coarse = len(mesher.mesh.quads)

    mesher._divisions_x.set("8")
    root.update()

    assert coarse == 12
    assert len(mesher.mesh.quads) == 24


def test_half_typed_input_is_reported_rather_than_raising(mesher, root) -> None:
    mesher._field_vars["length"].set("")
    root.update()

    assert "cannot mesh" in mesher.status_text
    assert mesher.mesh is None
    assert mesher._details.cget("text") == ""


def test_an_impossible_panel_is_reported(mesher, root) -> None:
    # A stiffener spaced wider than the panel has no plating to couple to.
    mesher._field_vars["spacing"].set("99000")
    root.update()

    assert "cannot mesh" in mesher.status_text
    assert mesher.mesh is None


def test_switching_shape_rebuilds_the_form(mesher, root) -> None:
    mesher._shape.set("plate")
    mesher._rebuild_fields()
    root.update()

    assert set(mesher._field_vars) == {"length", "width", "thickness"}
    assert mesher.mesh is not None
    assert not mesher.mesh.beams
    assert not mesher.mesh.couplings

    mesher._shape.set("beam")
    mesher._rebuild_fields()
    root.update()

    assert set(mesher._field_vars) == {"length"}
    assert mesher.mesh.beams
    assert not mesher.mesh.quads


def test_quadratic_and_alignment_checkboxes_reach_the_mesher(mesher, root) -> None:
    mesher._quadratic.set(True)
    mesher.refresh()
    root.update()

    assert mesher.mesh.is_quadratic
    assert all(len(nodes) == 8 for nodes in mesher.mesh.quads.values())

    mesher._align.set(True)
    mesher.refresh()
    root.update()
    assert mesher.mesh is not None


def test_a_stretched_panel_shows_the_quality_warning(mesher, root) -> None:
    mesher._field_vars["length"].set("40000")
    mesher._field_vars["width"].set("1000")
    # One stiffener at mid-width, so the panel is genuinely stretched rather than
    # having a stiffener hanging off the plate.
    mesher._field_vars["stiffeners"].set("1")
    mesher._field_vars["spacing"].set("500")
    mesher._divisions_x.set("2")
    mesher._divisions_y.set("2")
    root.update()

    assert mesher.mesh is not None
    assert verify_mesh_quality(mesher.mesh).warnings
    assert "aspect ratio" in mesher.status_text


def test_the_mesh_can_be_saved_and_reloaded(mesher, root, tmp_path) -> None:
    from anymesher.serialize import save_mesh

    root.update()
    mesh = mesher.mesh
    assert mesh is not None

    path = tmp_path / "from_gui.json"
    save_mesh(path, mesh)
    reloaded = load_mesh(path)

    assert reloaded.quads == mesh.quads
    assert reloaded.couplings == mesh.couplings


def test_embedded_mesher_applies_the_current_mesh_to_the_host(root) -> None:
    from anymesher.gui import MesherWindow

    received = []
    frame = MesherWindow(root, on_apply=received.append)
    frame.pack(fill="both", expand=True)
    root.update()

    frame.apply()

    assert received == [frame.mesh]
    assert frame.applied_mesh is frame.mesh
    frame.destroy()
    root.update()


def test_open_mesher_returns_a_hosted_mesh_picker(root) -> None:
    from anymesher.gui import open_mesher

    received = []
    window, frame = open_mesher(root, on_apply=received.append, title="Mesh picker")
    root.update()

    frame.apply()

    assert window.title() == "Mesh picker"
    assert frame.master is window
    assert received == [frame.mesh]
    window.destroy()
    root.update()


def test_the_window_tears_down_cleanly(root) -> None:
    # Widget attributes that collide with tkinter's own internals only fail on
    # destroy, and only sometimes, so the teardown path is asserted directly.
    from anymesher.gui import MesherWindow

    frame = MesherWindow(root)
    frame.pack(fill="both", expand=True)
    root.update()
    frame.destroy()
    root.update()

    assert not frame.winfo_exists()
