# Changelog

## Unreleased

## 0.5.0 - 2026-09-04

- Add immutable native-v2 options and serializable spatial metric controls while
  retaining the byte-compatible legacy lattice and metric defaults.
- Add deterministic SPD metric evaluation, physical/chart pullback, gradation
  limiting, and optional GIL-releasing C++17 kernels with Python oracles.
- Add an atomic mutable T3 topology, protected-edge contracts, component-owned
  shared seed identities, and deterministic canonical export.
- Add opt-in planar Frontal-Delaunay insertion with bounded queues, metric-aware
  off-centres, cancellation, geometry-limited diagnostics, and quality guarding.
- Add the frozen native-v2 baseline corpus and performance acceptance envelope.

## 0.4.0 - 2026-09-03

- Change the project license prospectively to the Mozilla Public License 2.0.
  Earlier published versions retain their historical license terms.
- Declare original project documentation under Creative Commons Attribution
  4.0 and add an explicit third-party dependency notice inventory.
- Add release-time assertions for the MPL SPDX expression and required license
  and notice files in source and binary distributions.

- Bind qualified triangle admission and deterministic bounded repair to the
  accepted S3 V2D formulation identity for the coordinated 0.3.2 activation
  candidate. Failed admission remains a typed failure with no legacy fallback.

- Score and refine native surface candidates against angle, scaled-Jacobian, and adjacent-element-growth limits in addition to aspect ratio, preventing quality-invalid slivers from being selected for automatic fallback.

- Propagate short imprint-edge spacing across arbitrary native face boundaries before triangulation, preventing acute and rapid-growth transitions without moving topology-owned nodes or relaxing quality policy.

- Allow automatic native fallback to refine edges from rejected structured faces instead of retaining their structured seed solution as hard overrides; mapped/native interface seeds remain locked and conformal.

- Preserve declared transverse plate-junction edges through upstream imprinting and structured-layout edge descendants so valid three- and four-shell junctions pass strict quality validation.

- Refine native seeding across thin four-sided imprint fragments so opposite intersection curves retain compatible divisions without relaxing structured quality limits.

- Preserve declared plate-junction incidence when meshing geometry that an upstream owner has already imprinted along an exact shared transverse boundary.

## 0.3.1 - 2026-08-27

- Permit structured shell edges with more than two attached shell elements only when they are exact node segments derived from an explicitly applied plate/plate intersection; undeclared non-manifold edges remain hard failures.
- Preserve declared plate-junction edge evidence through mesh serialization and add structured crossing-plate and fail-closed regressions.

## 0.3.0 - 2026-08-25

- Add a provider-neutral, strict JSON mesh command protocol with capability
  discovery, geometry selection, bounded mesh queries, revision-bound planning,
  atomic candidate publication, and typed failures.
- Add qualified commands for mesh controls, scope, edge divisions, local
  refinement, generation, and bounded in-memory undo/redo without exposing raw
  node or element mutation.
- Add a long-lived `anymesher automation` JSON Lines transport. Natural-language
  interpretation, model clients, credentials, network access, and filesystem
  paths remain outside command payloads.
- Require ANYgeometry 0.4 for its provider-neutral quantities, selectors,
  canonical entity handles, errors, and automation protocol primitives.

## 0.2.5 - 2026-08-22

- Add a bounded global structured-layout planner with exact shared-edge seed
  equations, mapped promotion, planar radial and O-grid partitions, immutable
  preview/application reports, and deterministic semantic hashes.
- Enforce the public `quality_v2` policy on the generated mesh. Explicit
  mapped requests now fail closed; automatic requests may use the recorded
  native fallback when the structured candidate violates quality or growth
  limits.
- Prepare plate/plate, beam/beam, and beam/shell connectivity on a detached
  geometry clone through ANYgeometry's public query/plan/apply contract.
  Coplanar positive-area overlaps are blocked until the user runs the explicit
  Fragment Overlaps geometry operation.
- Preserve face metadata, parameterization, exact topology lineage, edge seed
  intent, refinements, structural preparation provenance, and source-bound
  mesh associations across detached partitioning and meshing.
- Bound candidate, face, edge, block, and element work; add cooperative
  cancellation checkpoints; keep GUI, hardware, and long-running tests
  explicitly opt-in.
- This is the next public release after 0.2.3. Version 0.2.4 was used only as
  an internal integration milestone and was not published.

## 0.2.3 - 2026-08-21

- Replace the native rectangular interior-point grid with a deterministic
  staggered triangular lattice that stays clear of protected segments.
- Add bounded protected-edge flips, constrained smoothing, and at most two
  local refinement rounds while preserving boundary, shared, and explicit
  node coordinates.
- Report scaled Jacobian, angle, poor-element, point-budget, and per-face
  optimization diagnostics without rejecting an unavoidable valid mesh.

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
