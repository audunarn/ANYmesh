"""A tkinter mesher for the primitives.

Enter a panel, plate or beam, mesh it, look at it, read the quality report, save
it.  The preview is a plan-view projection on a plain ``Canvas``, so the editor
adds no dependency beyond the standard library.

Deliberately not a geometry editor.  Building a BRep interactively -- drawing
lines, splitting faces, punching holes -- is an application's job and ANYfem
already does it; duplicating it here would mean two editors drifting apart. What
this offers is the case a form actually suits: a shape defined by a dozen numbers.

Nothing here is imported by ``anymesher/__init__.py``, so importing the package
never requires a display or a tkinter build.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mesh import Mesh
from .primitives import (
    PanelMeshConfig,
    StiffenedPanel,
    StiffenerCrossSection,
    beam_mesh,
    panel_edge_nodes,
    simple_panel_mesh,
    stiffened_panel_mesh,
)
from .quality import verify_mesh_quality
from .serialize import save_mesh

__all__ = ["MesherWindow", "open_mesher", "main"]

_SHAPES = ("stiffened panel", "plate", "beam")

# (key, label, default).  Lengths are entered in mm because that is how plate and
# profile dimensions are quoted; the package is strictly SI in metres, so the
# conversion happens here at the widget and nowhere else.
_Field = Tuple[str, str, str]

_SHAPE_FIELDS: Dict[str, Tuple[_Field, ...]] = {
    "stiffened panel": (
        ("length", "Length [mm]", "4000"),
        ("width", "Width [mm]", "3000"),
        ("thickness", "Plate t [mm]", "12"),
        ("stiffeners", "Stiffeners [-]", "2"),
        ("spacing", "Spacing [mm]", "1000"),
        ("height", "Web height [mm]", "300"),
        ("web_thickness", "Web t [mm]", "10"),
        ("flange_width", "Flange b [mm]", "150"),
        ("flange_thickness", "Flange t [mm]", "15"),
    ),
    "plate": (
        ("length", "Length [mm]", "2000"),
        ("width", "Width [mm]", "1000"),
        ("thickness", "Plate t [mm]", "10"),
    ),
    "beam": (("length", "Length [mm]", "3000"),),
}

_MM = 1000.0

_STIFFENER_TYPES = ("T-bar", "L-bulb", "Angle", "Flatbar")


class MesherWindow(ttk.Frame):
    """The mesher, as a frame so it can be embedded as well as run alone."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_apply: Optional[Callable[[Mesh], None]] = None,
    ) -> None:
        super().__init__(master, padding=8)
        self._field_vars: Dict[str, tk.StringVar] = {}
        self._mesh: Optional[Mesh] = None
        self._applied_mesh: Optional[Mesh] = None
        self._message = ""
        self._on_apply = on_apply

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_form()
        self._build_report()
        self._rebuild_fields()

    # ------------------------------------------------------------------ form

    def _build_form(self) -> None:
        form = ttk.Frame(self)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        row = 0
        ttk.Label(form, text="Shape").grid(row=row, column=0, sticky="w")
        self._shape = tk.StringVar(value="stiffened panel")
        shape_box = ttk.Combobox(
            form, textvariable=self._shape, values=_SHAPES, state="readonly", width=18
        )
        shape_box.grid(row=row, column=1, sticky="ew", pady=2)
        shape_box.bind("<<ComboboxSelected>>", lambda _event: self._rebuild_fields())

        row += 1
        ttk.Label(form, text="Stiffener type").grid(row=row, column=0, sticky="w")
        self._stiffener_type = tk.StringVar(value="T-bar")
        type_box = ttk.Combobox(
            form,
            textvariable=self._stiffener_type,
            values=_STIFFENER_TYPES,
            state="readonly",
            width=18,
        )
        type_box.grid(row=row, column=1, sticky="ew", pady=2)
        type_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh())

        row += 1
        self._fields_frame = ttk.LabelFrame(form, text="Dimensions", padding=6)
        self._fields_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        self._fields_frame.columnconfigure(1, weight=1)

        row += 1
        divisions = ttk.LabelFrame(form, text="Divisions", padding=6)
        divisions.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        divisions.columnconfigure(1, weight=1)
        self._divisions_x = self._entry(divisions, 0, "Along length", "8")
        self._divisions_y = self._entry(divisions, 1, "Across width", "6")
        self._beam_divisions = self._entry(divisions, 2, "Along stiffener", "8")

        row += 1
        options = ttk.Frame(form)
        options.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        self._quadratic = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options, text="Quadratic (8-node)", variable=self._quadratic, command=self.refresh
        ).pack(anchor="w")
        self._align = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options, text="Align mesh to stiffeners", variable=self._align, command=self.refresh
        ).pack(anchor="w")

        row += 1
        buttons = ttk.Frame(form)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Save mesh...", command=self.save).pack(side="left")
        if self._on_apply is not None:
            ttk.Button(buttons, text="Use mesh", command=self.apply).pack(side="left", padx=4)

    def _entry(self, parent: tk.Misc, row: int, label: str, default: str) -> tk.StringVar:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        variable = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=row, column=1, sticky="ew", pady=1
        )
        variable.trace_add("write", lambda *_: self.refresh())
        return variable

    def _rebuild_fields(self) -> None:
        for child in self._fields_frame.winfo_children():
            child.destroy()
        self._field_vars = {}
        for index, (key, label, default) in enumerate(_SHAPE_FIELDS[self._shape.get()]):
            self._field_vars[key] = self._entry(self._fields_frame, index, label, default)
        self.refresh()

    # ---------------------------------------------------------------- report

    def _build_report(self) -> None:
        report = ttk.Frame(self)
        report.grid(row=0, column=1, sticky="nsew")
        report.columnconfigure(0, weight=1)
        report.rowconfigure(2, weight=1)

        self._status = ttk.Label(report, text="", anchor="w", wraplength=520, justify="left")
        self._status.grid(row=0, column=0, sticky="ew")

        self._details = ttk.Label(
            report, text="", anchor="w", justify="left", font=("TkFixedFont", 9)
        )
        self._details.grid(row=1, column=0, sticky="ew", pady=(6, 6))

        preview = ttk.LabelFrame(report, text="Plan view", padding=4)
        preview.grid(row=2, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(
            preview, width=520, height=320, background="white", highlightthickness=0
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", lambda _event: self._draw())

    # ------------------------------------------------------------- meshing

    @property
    def status_text(self) -> str:
        """The message currently shown to the user."""

        return self._message

    @property
    def mesh(self) -> Optional[Mesh]:
        """The mesh currently displayed, or ``None``."""

        return self._mesh

    @property
    def applied_mesh(self) -> Optional[Mesh]:
        """The last mesh sent to the embedding application, if any."""

        return self._applied_mesh

    def apply(self) -> None:
        """Send the current valid mesh to the embedding application."""

        if self._mesh is None:
            messagebox.showerror("Use mesh", "there is no valid mesh; fix the input first")
            return
        self._applied_mesh = self._mesh
        self.event_generate("<<MeshApplied>>")
        if self._on_apply is not None:
            self._on_apply(self._mesh)

    def _value(self, key: str) -> float:
        return float(self._field_vars[key].get())

    def build_mesh(self) -> Mesh:
        """Mesh whatever the form currently describes."""

        shape = self._shape.get()
        if shape == "beam":
            return beam_mesh(self._value("length") / _MM, int(self._divisions_x.get()))
        if shape == "plate":
            return simple_panel_mesh(
                self._value("length") / _MM,
                self._value("width") / _MM,
                self._value("thickness") / _MM,
                int(self._divisions_x.get()),
                int(self._divisions_y.get()),
                self._quadratic.get(),
            )
        panel = StiffenedPanel(
            length=self._value("length") / _MM,
            width=self._value("width") / _MM,
            plate_thickness=self._value("thickness") / _MM,
            stiffener_type=self._stiffener_type.get(),
            stiffener_spacing=self._value("spacing") / _MM,
            stiffener_height=self._value("height") / _MM,
            stiffener_web_thickness=self._value("web_thickness") / _MM,
            stiffener_flange_width=self._value("flange_width") / _MM,
            stiffener_flange_thickness=self._value("flange_thickness") / _MM,
            num_stiffeners=int(self._value("stiffeners")),
        )
        config = PanelMeshConfig(
            shell_num_divisions_x=int(self._divisions_x.get()),
            shell_num_divisions_y=int(self._divisions_y.get()),
            beam_num_divisions=int(self._beam_divisions.get()),
            use_8node_shells=self._quadratic.get(),
            align_mesh_to_stiffeners=self._align.get(),
        )
        return stiffened_panel_mesh(panel, config)

    def _set_status(self, message: str, colour: str) -> None:
        self._message = message
        self._status.configure(text=message, foreground=colour)

    def refresh(self) -> None:
        """Re-mesh and redraw.  Called on every edit."""

        self._mesh = None
        try:
            mesh = self.build_mesh()
        except (ValueError, KeyError, ZeroDivisionError) as error:
            # A half-typed number is the common case, so it reads as incomplete
            # input rather than as a failure.
            self._set_status(f"cannot mesh: {error}", "#8a5a00")
            self._details.configure(text="")
            self._draw()
            return

        self._mesh = mesh
        quality = verify_mesh_quality(mesh)
        if quality.warnings:
            self._set_status("; ".join(quality.warnings), "#8a5a00")
        else:
            self._set_status(
                f"{mesh.order} mesh: {mesh.num_nodes} nodes, {mesh.num_elements} elements",
                "#006000",
            )
        self._details.configure(text=self._details_text(mesh, quality))
        self._draw()

    def _details_text(self, mesh: Mesh, quality) -> str:
        lines = [
            f"nodes       {mesh.num_nodes}",
            f"quads       {len(mesh.quads)}",
            f"tris        {len(mesh.tris)}",
            f"beams       {len(mesh.beams)}",
            f"couplings   {len(mesh.couplings)}",
            f"aspect      max {quality.max_aspect_ratio:.3f}  mean {quality.mean_aspect_ratio:.3f}",
            f"warp        max {quality.max_warp:.5f}",
        ]
        if self._shape.get() == "stiffened panel":
            edges = panel_edge_nodes(mesh)
            lines.append(
                "edge nodes  " + "  ".join(f"{name} {len(edges[name])}" for name in ("x0", "xL", "y0", "yW"))
            )
            try:
                section = StiffenerCrossSection.from_geometry(
                    self._stiffener_type.get(),
                    self._value("height") / _MM,
                    self._value("web_thickness") / _MM,
                    self._value("flange_width") / _MM,
                    self._value("flange_thickness") / _MM,
                )
            except (ValueError, KeyError):
                pass
            else:
                lines.append(
                    f"section     A {section.area * 1.0e6:.0f} mm2  "
                    f"Iy {section.Iy * 1.0e12:.3g} mm4  Iz {section.Iz * 1.0e12:.3g} mm4"
                )
        return "\n".join(lines)

    # ----------------------------------------------------------------- plot

    def _draw(self) -> None:
        canvas = self._canvas
        canvas.delete("all")
        width = max(int(canvas.winfo_width()), 160)
        height = max(int(canvas.winfo_height()), 120)
        mesh = self._mesh
        if mesh is None or not mesh.nodes:
            canvas.create_text(width // 2, height // 2, text="nothing to show", fill="#909090")
            return

        positions = mesh.node_positions()
        # A plan view: x across, y up.  Enough to see division counts, stiffener
        # positions and whether the mesh is aligned -- which is what the numbers
        # in the form are actually about.
        x_min, x_max = float(positions[:, 0].min()), float(positions[:, 0].max())
        y_min, y_max = float(positions[:, 1].min()), float(positions[:, 1].max())
        span_x = max(x_max - x_min, 1.0e-9)
        span_y = max(y_max - y_min, 1.0e-9)
        margin = 16
        scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
        offset_x = (width - scale * span_x) / 2.0
        offset_y = (height - scale * span_y) / 2.0

        def to_pixel(node_id: int) -> Tuple[float, float]:
            position = mesh.nodes[node_id]
            x = offset_x + (float(position[0]) - x_min) * scale
            y = height - offset_y - (float(position[1]) - y_min) * scale
            return x, y

        for element_id in mesh.quads:
            points: List[float] = []
            for node in mesh.corners_of(element_id):
                points.extend(to_pixel(node))
            canvas.create_polygon(*points, outline="#4a6f9c", fill="", width=1)
        for element_id in mesh.tris:
            points = []
            for node in mesh.corners_of(element_id):
                points.extend(to_pixel(node))
            canvas.create_polygon(*points, outline="#4a6f9c", fill="", width=1)
        for connectivity in mesh.beams.values():
            start = to_pixel(connectivity[0])
            end = to_pixel(connectivity[-1])
            canvas.create_line(*start, *end, fill="#c04000", width=2)
        for coupling in mesh.couplings.values():
            x, y = to_pixel(coupling.beam_node)
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="#c04000", fill="#c04000")

    # ------------------------------------------------------------ file menu

    def save(self) -> None:
        if self._mesh is None:
            messagebox.showerror("Save failed", "there is no mesh to save; fix the input first")
            return
        path = filedialog.asksaveasfilename(
            title="Save mesh", defaultextension=".json", filetypes=[("Mesh JSON", "*.json")]
        )
        if not path:
            return
        try:
            save_mesh(path, self._mesh, overwrite=True)
        except OSError as error:
            messagebox.showerror("Save failed", str(error))


def open_mesher(
    master: tk.Misc,
    *,
    on_apply: Optional[Callable[[Mesh], None]] = None,
    title: str = "ANYmesher",
) -> Tuple[tk.Toplevel, MesherWindow]:
    """Open an embeddable mesher and return its window and frame."""

    window = tk.Toplevel(master)
    window.title(title)
    window.minsize(880, 560)
    mesher = MesherWindow(window, on_apply=on_apply)
    mesher.pack(fill="both", expand=True)
    return window, mesher


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Open the mesher."""

    root = tk.Tk()
    root.title("ANYmesher")
    root.minsize(880, 560)
    window = MesherWindow(root)
    window.pack(fill="both", expand=True)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
