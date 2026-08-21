# Changelog

## Unreleased

## 0.2.2 - 2026-08-21

- Require ANYgeometry 0.2.2 and consume its exact boundary-curve CONNECT for
  nonplanar extrusion walls on convex hole-free planar supports.
- Preserve the geometry-owned shared Edge, FaceUse, and Coedge identities
  through hybrid meshing without coordinate-inferred connection or healing.
- Keep unsupported partial, ambiguous, holed, nonconvex, and general
  nonplanar intersections fail-closed, with no backend-default or mesh-format
  change.

## 0.2.1 - 2026-08-21

- Require ANYgeometry 0.2.1 or newer within the qualified 0.2 line, pin the
  accepted geometry source in CI/release workflows, and add a disabled-native
  cell proving absence-only Python fallback and fail-hard explicit native use.
- Change the six native-triangulation public defaults to `auto` for 0.2.1 while
  preserving explicit `python`, fail-hard `native`, and Python fallback only
  when native capability is absent. ANYfem format 6 persists the selector so
  legacy projects remain explicitly Python-backed after migration.
- Depend on ANYgeometry as the single owner of `GeometryModel`, `EntityRef`,
  topology entities, curves, chain sampling and general geometry operations.
- Keep `anymesher.geometry` as exact-identity compatibility imports.
- Keep mapped-face checks, triangle-to-quad conversion and butterfly-hole
  decomposition in `anymesher.decomposition`.
- Preserve the historical mapped `split_face_at`, `split_face_between` and
  `strip_face` behavior in that module while ANYgeometry owns the neutral,
  general-purpose variants.
- Reject neutral triangular and trimmed faces at the mapped-backend boundary
  with a mesh-specific diagnostic instead of restricting neutral topology.
- Rebuild ANYgeometry Bezier splines exactly in the optional Gmsh backend and
  preserve edge association on their generated nodes. Exclude Gmsh's isolated
  circle-centre and spline-control construction nodes from the neutral mesh.
- Verify mapped and Gmsh remeshing from an ANYgeometry generator through owner
  replacement history and semantic groups.

Added:

- **Embeddable mesh selection.** `MesherWindow(on_apply=...)` adds a **Use mesh**
  button, and `open_mesher` opens the same live mesher inside a host Tk
  application.

## 0.1.0

First feature release. The geometry kernel and mapped mesher come from ANYfem,
the primitives from ANYsolver; see [MIGRATION.md](MIGRATION.md) for provenance.

Added:

- **Geometry** — vertices, edges carrying a straight or arc curve, four-sided
  faces, and the decomposition operations (`split_face_at`,
  `split_face_between`, `strip_face`, `triangle_to_quads`,
  `punch_circular_hole`, `check_mappable`).
- **Mapped meshing** — the transfinite Coons mesher, seeding with per-edge
  overrides, and local size-field refinement.
- **Neutral mesh** — `Mesh` with nodes, quads, tris, beams, coupling records and
  the geometry association; `Coupling`, which generalizes the old
  `(beam_node, plate_node)` pair to weighted interpolation over several plate
  nodes.
- **Primitives** — `stiffened_panel_mesh`, `simple_panel_mesh`, `beam_mesh`,
  `StiffenedPanel`, `PanelMeshConfig`, `StiffenerCrossSection`, and
  `panel_edge_nodes` for reading the four panel edges back.
- **Coupling** — Q4 and Q8 shape functions, a reusable structured cell index, and
  `locate_shell_element_at_xy`.
- **Quality** — `verify_mesh_quality` returning a `MeshQuality` record; aspect
  ratio and warp, advisory rather than enforced.
- **Backends** — `generate_mesh(..., backend=...)` dispatching to the built-in
  mapped mesher or to gmsh behind the `[gmsh]` extra.
- **Serialization** — `mesh_to_dict`/`mesh_from_dict`/`save_mesh`/`load_mesh`,
  round-tripping the association as well as the coordinates.
- **Mesher window** — a tkinter form for the primitives with live re-meshing, a
  plan-view preview on a plain `Canvas` and a quality report; entry point
  `anymesher-gui`.
- **CLI** — `anymesher panel|plate|beam|quality|backends`, each with `--json`.

Verified against both sources with both importable at once, and exact rather than
merely close:

- 18 mapped-mesher configurations against `anyfem` at
  `245b82ec68496fde1f8880c6a360f69973208bca` — node IDs and coordinates, quads,
  beams, offset nodes, the structured grid, every association map, the seeding
  divisions, the coupling pairs, and the text of every error raised. Plus the
  refinement path and the split, triangle and butterfly-hole decompositions.
- 96 stiffened-panel configurations against `anysolver` 0.1.3 at
  `8b4553cc680ff925df850e627165fc336615eaba`, over division counts, stiffener
  counts, beam divisions, Q4/Q8 and aligned/uniform transverse grids — node and
  element IDs, coordinates, connectivity, every coupling's masters, shape weights
  and eccentricity, the four panel edge node sets, and the quality report.
  Plus 4 plates, 3 beams and all 5 stiffener profile families.

Changed from the sources:

- **`Mesh.couplings` values are `Coupling` records, not `(beam, plate)` tuples.**
  The interpolated case needs several plate nodes with weights, which a pair
  cannot express. `Coupling.node_to_node` builds the single-master case and
  `coupling.plate_node` reads it back, so the ANYfem edit at strip time is one
  function in `solve/build.py` plus two test lines.
- **`Mesh` gained `tris` and `thickness_of_face`.** The mapped mesher never
  produces a triangle, so both stay empty for it; gmsh does, and dropping them to
  keep the container tidy would silently delete part of the mesh.
- **The chain-sampling helpers moved into the geometry package**
  (`anymesher.geometry.chains`). In ANYfem they lived with the mesher, which the
  geometry package then imported from — so geometry depended on the mesher while
  the mesher depended on geometry. The move removes that cycle with no behaviour
  change. `GeometryError` and `MeshError` moved to `anymesher.errors` for the
  same reason, and are still importable from where they were.
- **`GeometryModel.arc_frame(edge_id)` is now public.** A backend rebuilding the
  model in another kernel needs the circle, not just samples along it; reaching
  into `_arc_frame` from outside the module would have been a naming bug.
- **`PanelGeometry` and `MeshConfig` became `StiffenedPanel` and
  `PanelMeshConfig`**, and lost their material, support-condition and load fields.
  A material name is not a meshing decision and interpreting `"Integrated"` is
  structural convention, so both belong to the consumer. Renamed rather than
  trimmed in place, so a same-named class with different fields cannot be
  mistaken for the original.

Historical limitations in the 0.1.0 implementation, stated rather than worked
around:

- The gmsh backend meshes **planar** faces only, refuses arcs sweeping 180
  degrees or more (gmsh's built-in kernel cannot express one as a single circle
  arc), and does not support eccentric beam offsets. Each refusal names the
  mapped backend, which handles all three exactly.
- `GeometryModel.add_face` required four sides, so the gmsh backend could only
  mesh faces the mapped mesher also accepted. The Unreleased ANYgeometry
  extraction supersedes this restriction: neutral faces may have arbitrary
  valid loops and holes, while the mapped backend still requires mapped
  partitioning.
- Meshing a geometry model from the command line lacked an owner serialization
  format. ANYgeometry now owns that format; ANYmesher's CLI remains focused on
  mesh primitives and saved meshes, while the API accepts a shared model.

## 0.0.1

- Repository scaffolding: packaging metadata under the distribution name
  `ANYmesher`, CI across Python 3.11-3.14 on Windows and Linux, and the layering
  checks that keep the package a leaf of the dependency graph.
