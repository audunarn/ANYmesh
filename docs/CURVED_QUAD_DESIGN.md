# Curved and Higher-Order Quad Meshing — Design Record (CH0–CH2)

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
| Planar quadratic Q8/T6 (`order="quadratic"`, planar face) | **QUALIFIED at CH2** through post-topology promotion and strict CH1 validity. |
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

## CH1 implementation record — interpolation and strict element validity

CH1 adds the internal module `anymesher.quad.high_order`.  It does **not** open
any new public meshing route.  `generate_hybrid_mesh_result` keeps the CH0
planar-linear guard and `quality_v2` keeps its corner/skeleton semantics.

The shared kernel covers the four frozen shell families:

- Q4: four cyclic corners on `[-1,1]^2`;
- Q8: the same four corners followed by midsides 01, 12, 23, 30;
- T3: corners 0, 1, 2 on the reference triangle;
- T6: the same corners followed by midsides 01, 12, 20.

`shape_values` and `shape_gradients` use one ordering, and
`evaluate_mapping` returns the mapped points, both parametric derivatives, the
surface-Jacobian vector, its magnitude, signed Jacobian and a normalized
mapping-quality measure. `physical_area` uses bounded deterministic quadrature.
`geometry_error` and `normal_error` compare against caller-supplied reference
points/normals only; CH1 deliberately does not bind to an ANYgeometry owner.

### Strict signed-Jacobian certificate

`certify_mapping_validity` is deliberately stricter than point sampling.  The
signed Jacobian scalar is enclosed with Bernstein coefficients: tensor-product
bicubic bounds on the Q4/Q8 square and total-degree-two bounds on the T3/T6
triangle.  Positive Gauss or witness samples alone can never produce
`CERTIFIED_POSITIVE`.  Ambiguous patches are subdivided deterministically;
exhausting the bounded depth/subdivision budget gives `UNRESOLVED`.

`INVALID` requires an actually evaluated non-positive witness under the
scale-aware tolerance semantics.  The report's lower/upper values retain the
conservative whole-reference-element Bernstein envelope, while subdivision
leaf bounds provide the positivity proof.  Float64 reconstruction residuals
and a scale-aware roundoff guard are included so numerically uncertain
near-zero cases fail closed.

Reports are immutable and expose deterministic `to_dict()` output containing
JSON primitives only.  Certification accepts a deterministic cancellation
callback checked before work and at each active patch; a truthy callback raises
`HighOrderCertificationCancelled` without mutating element coordinates.

CH1 tests include genuinely non-affine curved Q8 and T6 mappings, independent
dense signed-Jacobian samples enclosed by the reported bounds, an
`UNRESOLVED` case that becomes certified with deeper subdivision, and hidden
Q8/T6 midside inversions with evaluated invalid witnesses.  The existing
`skeleton-only` `quality_v2` contract remains unchanged.

## CH2 implementation record — planar Q8/T6 promotion

CH2 explicitly re-opens the CH0 planar higher-order guard while keeping curved-surface quad-first closed. For an explicit planar quad-first request with `order="quadratic"`, the already-qualified linear Q4/T3 topology is generated first and then promoted atomically. No second topology engine is introduced.

Promotion allocates exactly one midside node for each unique final shell corner edge. Q4 connectivity becomes Q8 in the frozen corner-plus-cyclic-midside order and T3 becomes T6. Existing element IDs, face ownership, vertex ownership, corner IDs, and linear boundary-station node IDs are retained. `nodes_of_edge` is expanded in source-edge orientation so the original linear chain occupies every even position. Calling promotion on an already quadratic mesh is a no-op.

Geometry-owned boundary intervals are not promoted with raw 3D chord midpoints. The final linear `nodes_of_edge` chain identifies source-edge ownership; adjacent station parameters are recovered from ANYgeometry and the source edge is sampled at their parameter midpoint. This preserves exact analytic ownership, including the P03 radius-0.9 circular-hole boundary. Interior shell edges continue to use chord midpoints. One canonical undirected edge key owns each midside, including Q8/Q8 and Q8/T6 interfaces and reversed shared source edges.

Every proposed Q8/T6 is checked with the CH1 `certify_mapping_validity` kernel before publication. Exact curvature can make a coarse boundary-adjacent residual T6 non-positive even though its linear T3 was valid. CH2 therefore permits a bounded, deterministic repair only for an unprotected opposite corner of such a T6; protected source-boundary/station nodes never move, the move is capped at `0.5 * target_size`, all incident high-order elements must certify after the move, and the entire promotion remains staged until the final validity pass. P03 h=0.5 requires three such interior-corner repairs; the maximum measured displacement is 0.12266702535935825 m. P01, the residual trapezoid, P05, and P07 require no topology change, and P01 performs no corner repair at all.

Cancellation before the final publish checkpoint leaves the supplied linear mesh unchanged. Quadratic quad-first with beam/coupling content fails closed because CH2 does not qualify B3 ownership. Explicit curved-surface quad-first remains `QuadPublicUnsupported`; cylinder work begins in CH3. Q9 and orders above quadratic remain deferred. `quality_v2` retains its frozen skeleton-only semantics.

### CH2 measured qualification points

- P01 h=0.5: 273 linear nodes / 240 Q4 / 0 T3 -> 785 nodes / 240 Q8 / 0 T6, exactly 512 unique midsides, 64 exact source-boundary midsides, zero corner repairs.
- Residual trapezoid h=0.75: 40 linear nodes / 26 Q4 / 6 T3 -> 111 nodes / 26 Q8 / 6 T6, 71 unique midsides; all Q8/T6 certify positive and Q8/T6 interfaces reuse one midside ID.
- P03 circular hole h=0.5: 795 nodes / 233 Q8 / 10 T6, 519 unique midsides, 76 exact source-boundary midsides; all added hole-boundary midsides lie on the exact source arcs (radius 0.9 about `(3.2, 2.4)` within `1e-10`).
- P07 graded h=1.0: 275 nodes / 82 Q8 / 2 T6, 179 unique midsides, with the qualified graded linear corner topology preserved.

The promotion cost is O(unique final shell edges) for midside ownership/allocation plus a bounded curved-boundary T6 repair pass. There is no coordinate welding and no per-element source-curve ownership search after the boundary interval map is built.
## CH3 implementation record — owner-qualified cylindrical linear quad-first

CH3 opens the explicit quad-first public route for **linear cylindrical faces only**. It does not introduce a second topology engine. The existing owner-qualified cylindrical atlas/patch preparation remains the authority for face selection and chart binding, and the resulting physical metric chart is adapted to the same boundary registry, constrained T3 seed, Q4/Q5 planning, residual-front driver, validator, and publication pipeline used by the planar route.

`CylindricalQuadDomain` is the seed-facing adapter. It binds one source FaceUse to its qualified `CylindricalMetricChart`, stores boundary topology by exact source edge identity, projects source positions into circumferential-arc-length / axial physical chart coordinates, and lifts chart nodes back through the owner chart. Model UUID/revision and chart currency are checked before use. Owner atlas qualification is performed once by `_cylindrical_public.prepare_bindings`; the per-face domain does not duplicate that full-atlas proof in the hot path.

Boundary stations remain source-edge/station identities. `BoundaryStationRegistry` is shared across selected cylinder faces, so a physical shared seam receives one station chain and one set of published mesh-node IDs; no coordinate welding is used. `nodes_of_edge` remains in intrinsic source-edge direction. The source GeometryModel stays read-only.

The public pre-dispatch admits a `Cylinder` only for explicit `order="linear"` quad-first requests. Cylindrical `order="quadratic"` remains typed unsupported until CH4, while unqualified/general curved surfaces remain unsupported. Planar requests and the `quad_options=None` legacy sentinel are unchanged.

### CH3 measured qualification points

- One pi/4 sector at target size 0.5: 17 nodes / 8 Q4 / 2 T3, `N_eq=9`, with 14 front attempts and 8 guided accepts.
- The same sector at target size 0.25: 43 nodes / 30 Q4 / 2 T3, `N_eq=31`, with 36 front attempts and 30 guided accepts; target-size refinement therefore changes the genuine topology rather than only diagnostics.
- The full authored eight-sector ring at target size 0.5: 88 global nodes / 64 Q4 / 16 T3, `N_eq=72`. The physical seam shared by sectors 0 and 7 reuses the exact same source-station node IDs on both faces and repeat execution is deterministic.
- Every published shell node in the CH3 fixtures lies on the unit source cylinder within `1e-10`; source model state is unchanged. Cancellation during face seeding propagates without partial publication.

Diagnostics identify the route as `quad-first-cylindrical`, preserve the genuine front counters per face, and record `geometry_family_by_face`. CH3 does not open cones, ruled/Coons surfaces, cylindrical Q8/T6, Q9, or a new option schema.

## CH4 implementation record — cylindrical Q8/T6 promotion

CH4 re-opens `order="quadratic"` for the CH3 owner-qualified cylindrical quad-first route. The CH3 linear Q4/T3 topology remains authoritative; CH4 applies the same staged CH2 promotion, preserving element IDs, corner connectivity, face ownership, source-edge station nodes and source geometry.

Geometry-owned boundary intervals continue to use exact source-edge midpoint parameters. For non-boundary shell edges on a cylindrical face, the two linear endpoints are projected into that face's qualified physical `CylindricalQuadDomain`, averaged in circumferential-arc-length/axial chart coordinates, and lifted back through the owner chart. This places one canonical midside on the cylinder instead of using a 3D chord midpoint. Shared source seams are already owned by `nodes_of_edge` and therefore reuse one exact midside identity; there is no coordinate welding.

The promotion still runs atomically on a detached mesh and every final Q8/T6 must pass the CH1 strict `certify_mapping_validity` certificate before publication. Cancellation at `quad-first:quadratic-promotion-ready` leaves the source model unchanged and publishes no partial quadratic mesh. Planar CH2 behavior is unchanged; general curved faces, cones/ruled/Coons surfaces, Q9 and orders above quadratic remain outside CH4.

### CH4 measured qualification points

- One pi/4 sector, h=0.5: linear `17 nodes / 8 Q4 / 2 T3` -> quadratic `43 nodes / 8 Q8 / 2 T6`, exactly `26` added unique midsides.
- Same sector, h=0.25: linear `43 nodes / 30 Q4 / 2 T3` -> quadratic `117 nodes / 30 Q8 / 2 T6`, exactly `74` added unique midsides.
- Full eight-sector ring, h=0.5: linear `88 nodes / 64 Q4 / 16 T3` -> quadratic `256 nodes / 64 Q8 / 16 T6`, exactly `168` added unique midsides. The periodic physical source seam keeps one expanded station chain and all added seam midsides remain on the unit cylinder within `1e-10`.
- All qualified cylindrical Q8/T6 mappings are `CERTIFIED_POSITIVE`; repeat ring topology/counts are deterministic and source geometry is unchanged.

The new cylindrical interior midside work is O(unique final shell edges): each edge is owned once, projected/lifted through its already-prepared face domain once, and inserted through the existing canonical midside registry. No cylindrical topology engine is added.

## CH5 implementation record — high-order geometry provenance certificate

CH5 closes the CH0 high-order reporting seam without opening a new topology or geometry family. Every successful explicit quad-first quadratic promotion now publishes one immutable `HighOrderMeshCertificate` in `mesh.hybrid_diagnostics["high_order_geometry"]`; linear quad-first truthfully publishes `NOT_APPLICABLE` instead. The certificate is assembled from the exact staged Q8/T6 mesh only after the existing CH1 strict final validity pass and before the atomic publication checkpoint.

Each immutable `HighOrderGeometryReport` binds one source face to model UUID/revision, geometry family, chart type and a 3D chart-origin point. It records Q8/T6 and certification counts, boundary/interior projection methods, the number of interior midsides and the maximum geometry residual. Boundary/interface midside provenance is explicit: canonical source edge plus station interval, source-edge midpoint parameter, final midside node ID, owner residual and curvature class. Current supported source-curve classes are reported conservatively as `straight` for `Straight`, `analytic_curved` for `Arc`, and `sampled` for any other owner curve representation.

The global certificate checks face uniqueness, model/revision consistency, per-face versus global Q8/T6 counts, unique boundary midside identity, total-versus-boundary midside counts and the global residual envelope. A positive certificate cannot contain an invalid or incompletely certified face report. Conflicting provenance for a canonical source interval fails closed rather than being merged by coordinate proximity.
CH5 does not re-run or weaken the CH1 mapping proof: the report consumes the immediately preceding strict final Q8/T6 certification results and adds geometry-owner provenance over those same staged coordinates/connectivities. Cancellation at `quad-first:quadratic-promotion-ready` still occurs before any mesh mutation, so no partial quadratic topology or certificate can escape. Report construction is O(unique final shell edges + faces) and adds no topology engine, coordinate welding or `quality_v2` change.

### CH5 measured qualification points

- P01 h=0.5: `785 nodes / 240 Q8 / 0 T6`, 512 unique midsides and 64 unique source-boundary midsides; the planar chart origin is `(0,0,0)`, every source edge is classified `straight`, and the maximum reported geometry residual is exactly `0.0`.
- P03 circular-hole h=0.5: `795 nodes / 233 Q8 / 10 T6`, 519 unique midsides and 76 unique source-boundary midsides; reports contain both `straight` and `analytic_curved` source intervals and the maximum residual is `5.140558480280521e-13`.
- One cylindrical pi/4 sector h=0.5: `43 nodes / 8 Q8 / 2 T6`, 26 unique midsides and 14 unique source-boundary midsides; the physical owner-chart origin lifts to `(1,0,0)`, straight/analytic-curved source intervals are distinguished, and maximum residual is `2.854117595420714e-13`.
- Full eight-sector cylindrical ring h=0.5: `256 nodes / 64 Q8 / 16 T6`, 168 unique midsides, 72 unique canonical source-boundary/interface midsides and eight face reports; maximum residual is `2.8563285926006375e-13` with no duplicate canonical provenance.


## CH6 implementation record — quadratic quad-first B3 ownership

CH6 re-opens the CH2 fail-closed mixed shell/beam boundary for **straight** beam/member content on the planar explicit quadratic quad-first route. It does not add a beam topology engine. The public mixed route continues to build the accepted quad-first Q8/T6 shell mesh and the existing mapped quadratic beam slice separately, then merges them through the established structural pipeline.

The mapped beam implementation remains authoritative: a quadratic beam spans two linear stations and uses the intervening station as its third node, yielding B3 connectivity in frozen `start-mid-end` order. The middle node is the exact chord midpoint required by ANYsolver's straight-sided B3. Existing mapped rejection of curved quadratic beam edges remains in force; CH6 does not define a curved beam.

The old blanket public guard that rejected every quadratic quad-first request containing beam/member content is removed. The private `_promote_quad_first_quadratic` guard for arbitrary pre-existing B2 content remains unchanged: CH6 only re-opens the coherent public mixed route where the beam slice is generated as B3 from the outset.

Structural coupling is still owned by the existing merge/preparation pipeline. No hidden B2 element is published under `mesh.order="quadratic"`; all shell bodies are Q8/T6 and all beam bodies are B3. Source geometry remains read-only and cancellation before quadratic publication remains atomic. Cylindrical quadratic shell/beam ownership remains fail-closed for a later tranche; CH6 does not widen the CH4 cylinder route.

## CH7 implementation record — cylindrical quadratic B3 ownership

CH7 re-opens the final CH6 fail-closed boundary for **straight** B3 member content on the owner-qualified cylindrical quadratic quad-first route. The CH4 Q8/T6 shell topology and the CH6 mapped straight-sided B3 generator remain authoritative; CH7 adds no shell or beam topology engine.

The public blanket rejection of cylindrical quadratic shell/beam combinations is removed only for coherently generated straight member content. Curved quadratic beam edges remain rejected by the existing B3 preflight. Structural attachment resolution continues to search only the explicitly owned target-face elements.

An exact owner-surface attachment point does not generally lie exactly on the polynomial Q8/T6 interior map. CH7 therefore permits a second, quadratic-only BVH lookup when the existing base tolerance misses. The extra tolerance is derived from the largest midside-to-chord deviation of the allowed target-face Q8/T6 elements; linear meshes are unchanged and unrelated faces are never admitted. The resulting coupling retains the physical projection gap explicitly as eccentricity rather than moving the owner point or shell.

The qualified full eight-sector ring with one straight radial member publishes `261 nodes / 64 Q8 / 16 T6 / 2 B3 / 1 coupling`. B3 midsides remain exact endpoint chord midpoints, the centre member station lies exactly on the unit owner cylinder, the shell remains owner-bound, serialization is stable, repeat generation is deterministic, cancellation remains atomic, and source geometry remains read-only.
