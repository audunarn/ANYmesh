# Changelog

## Unreleased

- Declare threadpoolctl as the optional `diagnostics` extra for the per-worker
  thread-count report. Without it the report records the counts as unknown.
  The private authored-project adapter no longer imports ANYfem; its two ANYfem
  checks are passed in as a `ProjectReferenceScope`.

- Keep stations that lie exactly on a hull edge in the boundary used by the
  finite-hull completeness check (`_convex_hull(..., retain_collinear=True)`).
  The monotone chain had dropped them, so the Python backend could not mesh
  the thin hybrid triangle case. The quad-first MCF expectations in
  `tests/quad_first_planar/test_pq6_geometry_mcf.py` are re-baselined to the
  complete 81-triangle seed; the public area stays 8.5. Four tests that need an
  installed or pinned ANYgeometry wheel now skip with a stated reason when this
  environment imports a source checkout.

- Skip element candidates that cannot satisfy the parametric inversion when a
  curved attachment station lies on a linear host face, and cache face-local
  coordinates for one connectivity application. Results are identical; on the
  15 m barge single pass the connectivity stage drops by about a fifth
  (27.1 s -> 21.0 s wall, interleaved, identical mesh digest).

- Mesh a planar face in its physical chart whatever else the model contains.
  Previously a face that was the only face of its model was meshed in raw
  parameter units, so a plate with sides other than one metre was sized
  incorrectly (a 2 x 1 m plate had a mean edge of 0.29 m instead of 0.18 m as
  in a larger model). The face-count condition and its parallel-route
  workaround are removed. Unit-square plates and multi-face models are
  unchanged.

- Add opt-in component-parallel meshing: `generate_hybrid_mesh_result_parallel`
  meshes components that ANYgeometry's `plan_independent_components` certifies
  as independent in separate processes and joins them. Components travel as
  `ModelClosure` transport messages. Anything not certified (older ANYgeometry
  without the planner, a refused partition, one component, unsupported options)
  runs on the serial route, with the reason in
  `mesh.hybrid_diagnostics["parallel"]`. Node and element numbering differs from
  the serial run and no audit or structural-preparation report is produced.
  The route accepts `working_copy`, selections naming every face and member,
  `interactive` certification without a change set, and a precomputed seeding.
  `ParallelOptions.executor` accepts a warm spawn pool. See
  `docs/PARALLEL_MESHING_STUDY.md`.
- Compute the shell-node set once per connectivity pass instead of once per
  junction. Results are identical; the connectivity phase of a 48-member
  crossing grid drops from 5.7 s to 0.6 s.
- Speed up the experimental quad-first route and the shared pure-Python
  triangulator. Results are unchanged. Five whole-mesh scans are gone:
  - Bowyer-Watson now finds cavity candidates through a uniform grid instead
    of scanning every live triangle per inserted point, and it keeps
    triangles in reusable slots.
  - Constrained triangulation no longer rebuilds the full edge incidence for
    every segment that already is an edge.
  - PSLG segment splitting prefilters points by bounding box.
  - `QuadMeshState.cells_at` reads the node/edge maps instead of scanning
    every cell.
  - Seed-lattice containment (uniform, graded and adaptive) is decided for
    the whole candidate grid at once.

  On the 10 m x 6 m plate benchmark, 6,000 Q4 now take 4-5 s instead of
  12-16 s, and 12,298 Q4 take 10 s instead of 52 s. The adaptive layout at
  about 6,000 Q4, with or without a hole, drops from 19-24 s to 5-6 s.
  Scaling is now close to linear. The quad-first output is identical to 0.5.1
  on 86 planar, curved, refined, quadratic and adaptive cases.
- Speed up curved quad-first repair and quadratic promotion without changing
  results:
  - Face-chart dispatch resolves the optional `anygeometry.meshing` import
    and method signatures once per process, instead of a failed import
    search on every chart call.
  - High-order validity certification evaluates its Jacobian from exact
    single-point shape gradients and reuses repeated parameters.
- ANYmesher gains more from the next ANYgeometry release (0.4.5 candidate,
  `claude/projection-perf`), which makes projection, topology Coons
  evaluation and cylinder-atlas qualification faster with bit-identical
  results. No dependency change is needed; meshes are identical with 0.4.4.
  With both, the 72-case IS1 curved corpus drops from 104 s to 32 s, and
  meshing a hole-punched plate drops from 646 s to 165 s.

## 0.5.1 - 2026-09-28

- Fix mapped meshing of faces whose four declared sides fold under the
  transfinite map, such as an L-shaped plate with its re-entrant corner inside
  one mapped side. Through 0.5.0 the automatic strategy published such a mesh
  with nodes outside the face and overlapping elements, and native meshing of
  the same Plane face failed. The automatic strategy now meshes these faces
  natively, the explicit mapped strategy refuses them, and `check_mappable`
  reports the fold. Native meshing of a Plane face uses the plane's exact
  chart instead of the clipped four-corner patch. A face without a surface
  whose derived Coons surface folds now fails with an actionable error.
- Add an experimental, opt-in quad-first route (`QuadMeshingOptions`,
  `quad_options=` and `layout_policy=` on `generate_hybrid_mesh_result`).
  It covers planar faces with holes and concavity, owner-certified cylinders,
  analytic cones and ruled/Coons surfaces, linear Q4/T3 and quadratic Q8/T6
  output, and straight B3 members. It preserves exact source-edge station
  identity, transactional publication and cancellation. The default
  `quad_options=None` path is unchanged. The quad-first route was not part of
  the 0.5.0 release; its output and diagnostics may still change.
- Quad-first publishes only shape-admissible elements. Every Q4 needs a corner
  scaled Jacobian of at least 0.20, corners within 20-160 degrees and an edge
  ratio of at most 10. Residual triangles need a smallest angle of at least 15
  degrees. Tolerances are scale-free. Near-degenerate candidates stay
  triangles, residual triangles are improved by bounded diagonal flips, and a
  final element that still fails raises `QuadQualityRejected` instead of being
  published. Validation diagnostics report the worst element metrics and the
  active gates.
- Speed up the quad-first route and the shared pure-Python triangulator
  without changing any result. The Bowyer-Watson cavity search uses a
  conservative circumcircle pre-filter in front of the exact `incircle`
  predicate, ring containment is vectorized, the front driver keeps O(1)
  cell-kind counts, and seeding uses bucketed spacing checks. A 6,000-quad
  plate now takes about 31 s instead of several minutes. Quad-first seeds stay
  on the Python reference triangulation because the compiled backend breaks
  cocircular ties differently.
- Add `tools/build_quad_workers.py`, a cross-platform build for the optional
  quad-first MCF and TinyAD worker executables. The workers are not shipped in
  wheels; without them those stages report `UNAVAILABLE_SKIPPED`. CI builds them
  and runs the worker-dependent tests with `ANYMESHER_REQUIRE_QUAD_WORKERS=1`.
- Qualified-S3 admission now fails only below the solver's requirement. The
  new `S3_ADMISSION_FLOOR_POLICY` (smallest angle 15 degrees, normalized area
  0.30; other limits unchanged) is `DEFAULT_S3_QUALITY_POLICY`, backed by
  ANYsolver's reduced-angle qualification of the E4-PL S3 V2D element. The
  previous 30 degree envelope stays as `S3_TARGET_QUALITY_POLICY`: bounded
  repair and `prepare_qualified_s3_mesh` still work towards it and report any
  shortfall (`S3RepairResult.target_met`, record `quality_target`) instead of
  raising. Pass `quality_policy=S3_TARGET_QUALITY_POLICY` to keep the strict
  behavior. The production record is now
  `ANYMESHER_QUALIFIED_S3_PRODUCTION_PREPARATION_V2` and carries both
  policies. Shape limits are compared with a 1e-12 tolerance, so a triangle
  built exactly on a limit is no longer rejected by rounding.

## 0.5.0 - 2026-09-04

Alpha preparation: final source, platform, installed-wheel, performance and
hosted-CI qualification remains pending. This entry is not release approval.

- Add immutable native-v2 options and serializable spatial metric controls while
  retaining the byte-compatible legacy lattice and metric defaults.
- Add deterministic SPD metric evaluation, physical/chart pullback, gradation
  limiting, and optional GIL-releasing C++17 kernels with Python oracles.
- Add an atomic mutable T3 topology, protected-edge contracts, component-owned
  shared seed identities, and deterministic canonical export.
- Add opt-in planar Frontal-Delaunay insertion with bounded queues, metric-aware
  off-centres, cancellation, geometry-limited diagnostics, and quality guarding.
- Extend the opt-in route to owner-certified cylindrical patches and connected
  sector assemblies using physical chart lengths and topology-owned shared
  boundaries. Full-cylinder support uses connected sectors, not new single-face
  periodic topology. Other curved surface types remain outside this extension.
- Retain Python reference behavior while introducing persistent native topology
  storage and immutable local incidence, geometry, row and export caches.
  Keep protected identities, failed-candidate isolation and existing defaults.
- Add the frozen native-v2 baseline corpus and performance acceptance envelope.
- Require ANYgeometry 0.4.3 or newer within the qualified 0.4 line so clean
  production installs cannot resolve an older, unqualified geometry runtime.
- Defer field-guided quad-first meshing to a later release. Existing boundary
  collars and triangle recombination do not implement orientation-field solving
  or advancing quad fronts; no `field_guided_front` option is included.

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
