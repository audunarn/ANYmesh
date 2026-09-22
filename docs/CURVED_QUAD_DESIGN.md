# Curved and Higher-Order Quad Meshing — Design Record (CH0)

## Scope and intent

This record freezes the *contract surface* for curved surfaces and
higher-order (quadratic) elements on the quad-first route. CH0 is a
**contract-freeze** tranche: it establishes the design intent, the acceptance
contract, the re-use boundaries, and a small set of baseline-contract tests. CH0
is **not** an implementation tranche. No curved-surface topology engine, no
quadratic promotion path, and no new public driver are built here.

The planar programme (PQ0–PQ6) already qualified the genuine target-size planar
quad route through `generate_hybrid_mesh_result(..., quad_options=...)` with
`order="linear"` and `planar=True`. Everything CH0 adds is a declaration of what
is *frozen in place* versus what is *deferred*, plus the invariants that a later
CH1/CH2 implementation must not break.

## Fixed decisions

These choices are frozen for the curved/higher-order scope and are asserted by
the CH0 baseline-contract tests:

- **Q8 = 4 corners + 4 cyclic midsides.** The four corners keep the existing
  in-plane winding order, and the four midsides follow the same cyclic edge
  order. This is the serendipity node ordering that
  `shape_functions_8node` (src/anymesher/coupling.py:45) and the container
  contract at src/anymesher/mesh.py:111-112 already assume ("corners first,
  then the mid-side nodes").
- **T6 = 3 corners + 3 cyclic midsides.** Same convention for the triangle
  family (src/anymesher/mesh.py:114-118).
- **B3 = start–mid–end.** Beams keep the two endpoints and one midpoint in that
  order (src/anymesher/mesh.py:119-120).
- **Midside position is a geometry projection, not a raw chord midpoint, on a
  curved edge.** `insert_midside_nodes` (src/anymesher/surface_mesh.py:2060)
  accepts a `projector` callback for exactly this reason; the default
  chord-midpoint path is only valid where edges are straight.
- **`quad_options=None` remains the legacy dispatch sentinel.** Only an explicit
  `QuadMeshingOptions` (or mapping) signals the quad-first contract
  (src/anymesher/quad/options.py:87-100). CH0 adds no new sentinel.
- **Public order remains `linear` / `quadratic`.** `ELEMENT_ORDERS =
  ('linear', 'quadratic')`. There is no public "cubic" or "curved" order token.
- **There is exactly one topology engine.** CH0 must not introduce a second
  quad topology path; the curved-surface and higher-order scopes are
  extensions of the existing front/driver, not parallel generators.
- **Q9 is deferred.** No ninth-node or higher-order-quadrilateral contract is
  opened in CH0. The 8-node serendipity order is the highest-order quad the
  contract names.

## Environment facts (recorded, not assumed)

Facts below are frozen from the isolated curved-quad worktree and interpreter
metadata. No sibling repository source was inspected for CH0 evidence.

- Baseline: `a6c191466525b68b77f0a6cd5ad8bb4080b69243` on
  `opencode/curved-quad-v1`.
- Python `3.14.2`, numpy `2.4.6`, distribution `ANYmesher 0.5.0`.
- With the worktree `src` explicitly first on `sys.path`, `anymesher.__file__`
  is `C:\Github\ANYmesh\.worktrees\curved-quad-v1\src\anymesher\__init__.py`.
- Installed distribution `ANYgeometry 0.4.3`; interpreter-reported import path
  is `C:\Github\ANYgeometry\src\anygeometry\__init__.py`. This path is
  metadata only; CH0 did not inspect or read the sibling source tree.
- `anymesher._native` has no importable compiled extension in this new worktree.
  Native-parity tests therefore remain capability skips unless rebuilt locally.
- Q4 count/MCF worker was built locally from the committed checkout-independent
  script at `third_party/quad/worker/out/lemon/quad_mcf_worker.exe`, SHA-256
  `13fbb4c4961c0993bffba94584cdac687fc2cfc2d52a5415ea719588fce41d48`.
- Q5 TinyAD worker was built locally from the committed checkout-independent
  script at `third_party/quad/worker/out/tinyad/quad_tinyad_optimizer.exe`,
  SHA-256 `02e25212cd0b0cd53c96f0f4231eacbec5b57579704bc8255e2f76cae13b9329`.
  Both executables are ignored build outputs, not tracked CH0 changes.
- The primary checkout remains an unrelated dirty worktree. CH0 does not use its
  files or worker outputs and does not modify it.

## Reuse / extend / do-not-reuse matrix

Symbols named below are the **only** curved/higher-order seams in scope. A
later tranche must re-use them, not fork them.

### Re-use (already correct, do not touch)

| Symbol | Location | Role |
| --- | --- | --- |
| `shape_functions_8node(xi, eta)` | src/anymesher/coupling.py:45 | Canonical Q8 serendipity order; partition-of-unity reference. |
| `shape_functions_4node(xi, eta)` | src/anymesher/coupling.py:31 | Bilinear reference. |
| `insert_midside_nodes(mesh, *, projector=None)` | src/anymesher/surface_mesh.py:2060 | T3/Q4 -> T6/Q8 promotion; `projector` is the curved-edge seam. |
| `synchronize_existing_midpoints(source, working, stage, registry, ...)` | src/anymesher/_quadratic_boundary_prepare.py:8 | Reconciles pre-existing midside nodes on a shared stage. |
| `Mesh.is_quadratic` | src/anymesher/mesh.py:172 | `order == "quadratic"` flag. |
| `Mesh.corners_of(element_id)` | src/anymesher/mesh.py:189 | Corner-only view, order-independent. |
| `corner_edges(connectivity)` | src/anymesher/core.py:106 | Undirected corner edges for T3/T6/Q4/Q8 (midside-edge source). |
| `triangle_quality(points, triangles, ...)` | src/anymesher/quality_v2.py:142 | Quality slice of `connectivity[:, :3]` (skeleton-only). |
| `quad_quality(points, quads, ...)` | src/anymesher/quality_v2.py:220 | Quality slice of `connectivity[:, :4]` (skeleton-only). |
| `quad_candidate_quality(points, connectivity)` | src/anymesher/quality_v2.py:209 | Four-corner candidate metric helper. |

### Extend (curved/higher-order work happens here)

| Symbol | Location | Extension intent |
| --- | --- | --- |
| `CylindricalMetricChart` | src/anymesher/_cylindrical_chart.py:27 | Analytic chart for cylindrical metric; the curved-edge `projector` for midsides is derived from this, not hard-coded. |
| `prepare_cylindrical_patch(geometry, ...)` | src/anymesher/_cylindrical_patch.py:67 | Existing owner-qualified curved-patch seam reserved for CH3 cylinder work. |
| `prepare_cylindrical_atlas(geometry, ...)` | src/anymesher/_cylindrical_atlas.py:99 | Existing periodic/component seam reserved for CH3 cylinder work. |
| `QuadraticComponentStage.promote` | src/anymesher/_cylindrical_quadratic.py:42 | Component-scoped promotion stage; the single promotion site. |
| `finish_quadratic_components(geometry, mesh, bindings, boundary_registry)` | src/anymesher/_cylindrical_public.py:118 | Public assembly for quadratic components after promotion. |
| `FaceChart` | src/anymesher/charts.py | Chart abstraction every curved face must go through; no ad-hoc 2D frame per face. |

### Do-not-reuse (out of scope, never a curved/HO seam)

| Symbol | Reason |
| --- | --- |
| `native_constrained_smoothing`, `native_local_edge_flip`, `recombination_decisions` (native_cpp) | Legacy native topological operators; CH0 must not route curved/HO topology through the deprecated native recombine path. |
| `QuadPublicUnsupported` raise sites that hard-reject non-planar faces (src/anymesher/hybrid.py:1872, 2824, 2835, 2852) | These are the *current* planar guards. CH0 keeps them in place so the public route stays planar-only until a dedicated curved tranche explicitly opens; they are the boundary, not a helper to bypass. |
| `QuadMeshingOptions` schema `anymesher.quad-meshing-options/1` (src/anymesher/quad/options.py:26) | The option schema is frozen; CH0 adds no new option field for "curved" or "order=8". A new schema version is a separate tranche. |

## Proposed internal seams (frozen as names, not implemented)

CH0 freezes three *names* so a later tranche cannot invent divergent ones. None
exist yet; CH1 owns interpolation/validity seams and CH2 owns planar Q8/T6 enrichment. Cylinder use begins in CH3.

1. **`HigherOrderEdgeRegistry`** — maps each canonical interface/corner edge (see
   `canonical_interface_edge`, src/anymesher/quad/public_integration.py:42) to
   exactly one midside node, so shared curved edges do not grow two midsides.
   This is the single ownership authority for midside identity on the curved
   route and supersedes ad-hoc per-face midpoint allocation.
2. **`EnrichmentPlan`** — an explicit, inspectable declaration of which edges of
   a face receive midsides and at what parameter, produced before any node is
   inserted. Promotion (`QuadraticComponentStage.promote`) must consume an
   `EnrichmentPlan`, not infer midside placement from element geometry at
   publish time.
3. **`HighOrderGeometryReport`** — a per-face report that records the chart
   origin (`FaceChart`), the edge curvature class (straight / analytic curved /
   sampled), the midside projection method used, and the residual distance of
   each midside from its source curve. This is the certificate a later global
   high-order validity check reads.

## Public status matrix

| Capability | Public status at CH0 |
| --- | --- |
| Planar linear quad-first (order=`linear`, `planar=True`) | **QUALIFIED** through PQ6; the accepted public route. |
| Planar quadratic / curved (planar face, higher-order order token) | **QUAD-FIRST MISSING** — guard stays; no public driver. |
| Curved-surface quad-first (cylinder, cone, general curved face) | **QUAD-FIRST MISSING** — `QuadPublicUnsupported("...requires a planar face")` raised; no public driver. |
| Higher-order (Q8/T6/B3) quad-first | **QUAD-FIRST MISSING** — `ELEMENT_ORDERS` accepts `quadratic` but the quad-first public dataflow does not promote; container and shape functions are ready. |
| Q9 / non-serendipity / cubic order | **NO CONTRACT** — deferred; not named, not rejected, simply absent. |
| Global high-order validity certificate | **MISSING** — `HighOrderGeometryReport` is frozen as a name only; no global certificate emitted. |

## Crosswalk: PQ invariants a curved/HO tranche must not break

These hold for planar and must hold again once curvature/higher-order land:

- `quad_options=None` still executes the legacy path; an explicit instance is
  the only quad-first trigger.
- The public route stays self-contained and truthful: a missing worker is
  `QuadCapabilityMissing` / `UNAVAILABLE_SKIPPED`, never a silent fallback to a
  legacy route.
- Source geometry identity, revision, vertices, edges and face topology are
  unchanged by meshing under `GeometryMutationPolicy.READ_ONLY`.
- Promotion is coordinate-only on the topology: `QuadraticComponentStage.promote`
  adds midside nodes and rewrites connectivity width (3->6, 4->8) but must not
  re-tile corners, reseed the front, or move a protected boundary/station node.
- Canonical boundary/station ownership is invariant across the straight/curved
  split: the same edge identity that `BoundaryStationRegistry` owns must own the
  midside, in both the linear and the promoted mesh.

## Performance audit notes (frozen intent)

- Midside projection must be O(edges) per face, not O(elements x edges): derive
  one `projector` from the face `FaceChart` and reuse it across that face's
  midsides. The PQ4a review already removed a redundant O(N^2) duplicate scan in
  `_interior_lattice`; the curved tranche must not reintroduce per-edge chart
  rebuilds.
- `HigherOrderEdgeRegistry` must be shared across faces that share an edge so a
  curved edge inserted twice is detected as a single canonical identity, not two
  nodes. This mirrors the `BoundaryStationRegistry` shared-edge contract proven
  in P09.
- The high-order quality slice stays skeleton-only (`[:, :4]` / `[:, :3]`);
  adding a warped/curved quality metric is a *separate* report, not a change to
  the frozen `quad_quality` / `triangle_quality` signatures.

## Deferred: Q9 and higher orders

CH0 explicitly does not name any order beyond Q8 serendipity. There is no
`Order9`, no full-function cubic quad, and no `quad_options` field selecting one.
`ELEMENT_ORDERS` remains `('linear', 'quadratic')`. If a future tranche opens a
higher order, it must introduce a new schema version of
`QuadMeshingOptions` and a new shape-function reference, and must not widen the
frozen Q8 contract here.

## Global high-order validity certificate — status

**Missing, by design, at CH0.** The name `HighOrderGeometryReport` is frozen so
that when it is implemented there is exactly one certificate type, but no global
"this mesh is a valid high-order curved mesh" aggregate is emitted or asserted at
CH0. The per-face `HighOrderGeometryReport` (seam above) is the intended unit; a
cross-face validity certificate is a later tranche.

## CH0 gate contract (summary; see acceptance doc for assertions)

The CH0 baseline-contract tests assert the frozen facts above from source alone —
ordering, the `QuadPublicUnsupported` planar guard, `None` sentinel, container
Q8/T6/B3 widths, and that no curved/HO public route accidentally exists yet. They
are documentation-as-code: they go green at CH0 and must stay green after CH1/CH2
unless the corresponding freeze is explicitly re-opened.
