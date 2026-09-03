# ANYmesher

Structured meshing for shell and beam finite-element models described by
[ANYgeometry](https://github.com/audunarn/ANYgeometry): edge seeding and local
refinement, a built-in mapped (transfinite Coons) mesher, mapped-face
decomposition, optional Gmsh meshing, geometry-to-mesh associations, quality
metrics, a tkinter mesher and a command-line interface.

Install the released package with `python -m pip install ANYmesher`. For local
development, use the editable setup below.

The repository is `ANYmesh`, but `anymesh` was already taken on PyPI, so the
distribution is **`ANYmesher`** and the import package is **`anymesher`**.

## Quick start

A stiffened panel needs no geometry model:

```python
import anymesher as am

panel = am.StiffenedPanel(
    length=4.0, width=3.0, plate_thickness=0.012,
    num_stiffeners=2, stiffener_spacing=1.0,
    stiffener_height=0.30, stiffener_web_thickness=0.010,
    stiffener_flange_width=0.150, stiffener_flange_thickness=0.015,
)
mesh = am.stiffened_panel_mesh(panel, am.PanelMeshConfig(
    shell_num_divisions_x=8, shell_num_divisions_y=6, beam_num_divisions=8,
))

mesh.num_nodes, len(mesh.quads), len(mesh.beams), len(mesh.couplings)
am.panel_edge_nodes(mesh)["x0"]          # the nodes on the x = 0 edge
am.verify_mesh_quality(mesh).max_aspect_ratio
```

Anything less regular goes through the shared geometry model:

```python
import anygeometry as ag

model = ag.GeometryModel()
points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
face = model.add_face(model.add_polyline(points, close=True))

am.punch_circular_hole(model, face, (1.0, 0.5, 0.0), 0.2)
mesh = am.generate_mesh(model, target_size=0.05, order="quadratic")
mesh.nodes_on(ag.EntityRef("edge", 3))   # still addressable after re-meshing
```

`anymesher.geometry` remains as a temporary compatibility import. Its classes
are the exact ANYgeometry classes, not converted copies, but new code should
import geometry directly from `anygeometry`.

## Command line

```bash
anymesher panel --length 4 --width 3 --thickness 0.012 --height 0.3 --web-thickness 0.01 --divisions-x 8 --divisions-y 6
```

`panel`, `plate`, `beam`, `quality` and `backends`, each with `--json`.
`--output` writes the mesh as JSON.

### Provider-neutral automation

`anymesher automation --geometry model.json` starts a JSON Lines session for
LLMs and agent frameworks. The core accepts strict structured tools only; it has
no model SDK, prompt parser, credentials, network access, or arbitrary shell or
filesystem command. Discover the exact schemas with `mesh_capabilities`, plan a
revision-bound batch with `plan_mesh`, inspect the detached candidate, and
publish it exactly once with `apply_mesh`.

Qualified commands configure meshing, select geometry scope, pin edge divisions,
manage named refinements, generate, undo, and redo. Direct node/element mutation
is intentionally unavailable because it would invalidate topology associations
and deterministic numbering. Geometry edits remain in ANYgeometry's own
automation protocol.

`anymesher-gui` opens the mesher: enter a panel, plate or beam, watch it re-mesh
as you type, read the quality report, and save the result. It is deliberately not
a geometry editor — building a BRep interactively is an application's job, and
[ANYfem](https://github.com/audunarn/ANYfem) already does it.

Applications can put that form behind their own mesh button.  Passing a callback
adds a **Use mesh** button and returns the neutral `Mesh` directly:

```python
from anymesher.gui import open_mesher

window, mesher = open_mesher(root, on_apply=project.replace_mesh)
```

The host remains responsible for deciding whether a generic neutral mesh is
appropriate for its structural model.

## What a mesh is here

`generate_mesh` returns nodes, quadrilaterals, triangles, beams, coupling
records, and the **association** back to the geometry: which node came from which
vertex, which nodes lie along which edge, which elements belong to which face.

The association is the point. It is what lets a load or a restraint be named
against geometry and survive a re-mesh, and what makes results addressable by the
thing the user drew rather than by node number. Primitives fill the same fields
against synthetic entity IDs, so a consumer needs one lookup path either way.

## Backends

| | mapped (built in) | gmsh (`[gmsh]` extra) |
| --- | --- | --- |
| Structure | structured grid per face | unstructured |
| `grid_of_face` | filled | empty |
| Conformity | by construction | by gmsh |
| Elements | quads only | quads, with triangles where recombination cannot pair them |
| Curved faces | meshed exactly | refused — planar faces only |
| Eccentric beams | supported | not supported |

```python
am.generate_mesh(model, backend="gmsh", target_size=0.1)
```

The two are not interchangeable in what they guarantee, and the differences are
recorded rather than smoothed over: an unstructured mesh has no `(i, j)` index, so
that field is left empty instead of filled with something plausible.
Planar boundaries may contain ANYgeometry straight lines, circular arcs or
Bezier splines; Gmsh receives each as its corresponding exact curve primitive.

### Hybrid strategy: Automatic, Mapped, or Native

`generate_hybrid_mesh_result` is the production shell/beam route. The
`strategy` argument is explicit and is independent of the lower-level
triangulator selector:

| `strategy` | Behaviour |
| --- | --- |
| `"auto"` | Use mapped blocks where qualified; use native faces for the remainder. With `structured_options`, try the global structured plan first and use the recorded native fallback only when its actual mesh fails `quality_v2`. |
| `"mapped"` | Require every selected face to become a qualified mapped block. Any unsupported partition or quality violation blocks the mesh. |
| `"native"` | Require the unstructured face route. It cannot be combined with `structured_options`. |

```python
result = am.generate_hybrid_mesh_result(
    model,
    target_size=0.1,
    strategy="auto",                 # Automatic
    structured_options={},           # enable global structured planning
)

mapped = am.generate_hybrid_mesh(
    model,
    target_size=0.1,
    strategy="mapped",               # fail closed unless all faces map
    structured_options={},
)

native = am.generate_hybrid_mesh(
    model,
    target_size=0.1,
    strategy="native",
    native_backend="auto",           # compiled triangulator, else Python
)
```

Every call creates a detached structural closure. On that working copy,
ANYmesher uses ANYgeometry's exact public intersection query/plan/apply
workflow to connect crossing plates, beam ends and crossings, and beam/shell
intersections. The editable source model is unchanged; published mesh
associations are remapped to its original entity handles. Passing
`structural_preparation=False` disables automatic relationship creation but
still uses a detached clone.

Positive-area coplanar plate overlaps are never assigned to one plate
implicitly. Meshing blocks with a diagnostic directing the user to the
previewable, undoable **Fragment Overlaps** geometry command.

`plan_structured_layout` and `apply_structured_layout` are public detached
preview/application APIs. `commit_structured_layout` is feature-gated: it is
available only when the installed ANYgeometry exposes the exact
`FeatureHistory.adopt_frozen` contract. The qualified base dependency remains
ANYgeometry 0.2.2, so ordinary mesh generation never depends on that optional
feature-history operation.

### Native triangulation

Mapped/native face selection is separate from the triangulator used after a face
has selected the native strategy. In 0.2.1, omitted `native_backend` and planar
`backend` arguments default to `"auto"`: an explicitly registered boundary is
preferred, then the built-in compiled boundary, with the deterministic Python
reference used only when native capability is absent.

Use `"python"` to require the compatibility/reference implementation and
`"native"` to require compiled capability without fallback. Import or execution
failure from a present but corrupt extension is reported; it is never treated as
ordinary absence. ANYfem format 6 persists this selector explicitly, while its
format 1-5 migration writes `"python"` so old projects retain their behavior.

## Design notes

**One shared geometry authority.** ANYmesher never converts a geometry model.
It consumes the same `GeometryModel` and `EntityRef` objects that applications
use for selections and attributes, then records associations to their stable IDs.

**Conformity by construction.** Node generation order is fixed: one node per used
vertex, then `n - 1` interior nodes per edge in the edge's own direction, then face
interior nodes from the Coons blend. Neighbouring faces look their boundary nodes
up from the same registries, so they share the very same nodes. The alternative —
meshing faces independently and merging coincident nodes within a tolerance —
fails quietly on nearly-coincident geometry and produces a mesh that looks
connected and is not.

**Coupling records, not constraints.** When a stiffener stands off the plating,
the mesher records per beam node: the shell element it projects into, the shape
weights at that point, and the eccentricity vector. That is a statement about
geometry. Deciding it becomes six multi-point constraints is the consuming
solver's business. Interpolating through the shape functions is what removes the
older requirement that a beam node lie on a shell node row, so the mesh no longer
has to be aligned to the stiffeners for the coupling to be exact.

**Numbering is a contract.** A consumer stores results per node and element ID, so
a renumbering that is mathematically irrelevant still invalidates its baselines.
The primitives number shell nodes from 1, beam nodes from 10000, beam elements
from 20000 and couplings from 30000; the mapped mesher numbers by registry order.
Both are documented where they are implemented and asserted by test.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layering, and
[MIGRATION.md](MIGRATION.md) for what was extracted from where.

## Scope

Out of scope: geometry ownership, elements, materials, assembly and solution.
General splitting, trimming, projection, transforms and intersections belong to
ANYgeometry. `check_mappable`, `triangle_to_quads` and the four-patch butterfly
hole decomposition stay here because they exist specifically for mapped quads.
The production `generate_hybrid_mesh` workflow uses ANYgeometry's qualified
intersection operations on an immutable working closure; it does not introduce
a second geometry kernel or weld nodes by proximity. The older
`generate_mesh_with_intersections` name remains only as a deprecated migration
seam for historical models and is never selected automatically.
The legacy `anymesher.split_face_at`, `split_face_between` and `strip_face`
imports likewise retain their mapped-partition semantics; new neutral geometry
code should import the general edit operations from `anygeometry`.
Writing a mesh to a `.fem`
or `.inp` file belongs to [ANYfileio](https://github.com/audunarn/ANYfileIO), which
depends on this package — so the arrow cannot point back. The JSON in
`anymesher.serialize` is the mesh container written out as itself, not an
interchange format.

## License

Starting with version 0.4.0, ANYmesher source code is licensed under the
Mozilla Public License 2.0. See `LICENSE` for the full terms and `NOTICE` for
the prospective relicensing statement. Earlier published versions remain
available under the license terms that applied to those versions.

Original project documentation under `docs/` is licensed under Creative
Commons Attribution 4.0 as described in `docs/LICENSE.md`. Dependency and
optional-tool licenses are recorded in `THIRD_PARTY_NOTICES.md`.

## Units

SI throughout, lengths in m. There is no conversion layer. The mesher window
accepts mm because that is how plate and profile dimensions are quoted, and
converts at the widget.

## Development

```powershell
python -m pip install --no-deps -e C:\Github\ANYgeometry
python -m pip install -e "C:\Github\ANYmesh[dev,gmsh]"
python -m pytest
```

To open the mesher straight from a checkout — including an IDE's Run button, with
nothing installed — run [`run_gui.py`](run_gui.py) at the repository root.

```bash
python run_gui.py
```
