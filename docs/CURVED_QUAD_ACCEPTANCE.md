# Curved and Higher-Order Quad Meshing — Acceptance Record (CH0–CH2)

This file records the invariants that the CH0 baseline-contract test suite
`tests/quad_first_curved/test_ch0_baseline_contract.py` asserts. The suite
contains **exactly nine tests**, one per criterion below, each named after the
criterion. Tests are documentation-as-code: they pass at CH0 and must **remain**
passing after CH1 (interpolation/high-order validity) and CH2 (planar Q8/T6
promotion) unless the specific freeze is explicitly re-opened in a later design revision.

CH0 is a **contract-freeze** tranche. It adds no public API, no curved-surface
driver, and no quadratic promotion path. It only pins the facts that a later
implementation must honour. Every assertion is made from source already present
in the package; no worker binary and no compiled native module is required.

---

## Test 1 — `test_node_ordering_is_corners_then_cyclic_midsides`

The higher-order node order is frozen:

- **Q8 = 4 corners + 4 cyclic midsides.** Evaluated on the reference square
  `[-1, 1]^2` with the in-plane winding corner 0 at `(-1,-1)`, corner 1 at
  `(1,-1)`, corner 2 at `(1,1)`, corner 3 at `(-1,1)`, and midsides in the
  cyclic edge order (edge 01 `(0,-1)`, edge 12 `(1,0)`, edge 23 `(0,1)`, edge 30
  `(-1,0)`), `shape_functions_8node` returns `1.0` (within `1e-12`) at index
  `k` and `0.0` at every other index for the `k`-th node. That pins nodes 0–3
  as the corners in winding order and nodes 4–7 as the midsides in cyclic edge
  order (01, 12, 23, 30).
- **T6 = 3 corners + 3 cyclic midsides.** The container contract
  (src/anymesher/mesh.py) and `_boundary_sequence` name 6-node triangles as
  corners (0-1-2) then midsides (01, 12, 20).
- **B3 = start–mid–end.** Beams keep endpoints at indices 0 and 2, midpoint at 1.

`shape_functions_8node` is the numeric authority; the T6/B3 clauses are asserted
against the frozen container contract, since triangles and beams have no shared
2D basis reference.

## Test 2 — `test_promotion_maps_linear_topology_to_quadratic`

Promotion must be **coordinate-only and topology-preserving**:

- `insert_midside_nodes(mesh, projector=...)` (src/anymesher/surface_mesh.py:2060)
  on a linear quad mesh returns a new container whose per-element width is 8
  (was 4), whose `order` flag is `"quadratic"`
  (`Mesh.is_quadratic`, src/anymesher/mesh.py:172), and whose first four node
  ids of each promoted element are unchanged.
- Calling it again on the already-promoted container is idempotent: no new midside
  nodes are added and the width stays 8 (a promoted element is not re-promoted).
- The shared-edge invariant holds: the four midsides are derived from the four
  undirected corner edges returned by `corner_edges`
  (src/anymesher/core.py:106), so a curved edge shared by two faces cannot grow
  two midside nodes — one canonical edge, one midside.

## Test 3 — `test_shape_functions_form_a_partition_of_unity`

At every point `(xi, eta)` in the closed reference square `[-1, 1]^2`,
`sum_{k=0}^{7} N_k(xi, eta) == 1` within a tolerance of `1e-12`. This is a
fundamental property of the Q8 serendipity basis and must hold for the existing
implementation at src/anymesher/coupling.py:45. The test evaluates on a
13×13 uniform grid over the closed square (including edge and corner points,
where the sum is still exactly 1 by the serendipity property).

## Test 4 — `test_quality_mechanism_is_skeleton_only`

Quality is frozen to the **geometric skeleton** regardless of element order:

- `triangle_quality` (src/anymesher/quality_v2.py:142) slices the connectivity
  to `[:, :3]`. Moving the T6 midsides (columns 3:6) must leave the per-element
  quality vector unchanged for a fixed corner set.
- `quad_quality` (src/anymesher/quality_v2.py:220) slices to `[:, :4]`. Moving
  the Q8 midsides (columns 4:8) must leave the per-element quality vector
  unchanged for a fixed corner set.

The test drives this directly: it builds one element with fixed corners, makes
a second element whose corner columns are identical but whose higher-order
columns differ, and asserts the quality outputs are equal. This pins the
skeleton-only contract — a curved or warped quality metric is a *separate*
report type, not a change to these signatures.

## Test 5 — `test_public_symbols_resolve_at_top_level`

The curved/higher-order contract must not depend on a new import surface. Every
symbol the later tranches are allowed to re-use must resolve today through the
documented paths, so a later tranche cannot silently fork:

Top-level and submodule documented paths (verified on this checkout):

- `anymesher.shape_functions_8node`, `anymesher.shape_functions_4node`
- `anymesher.Mesh`, `anymesher.ELEMENT_ORDERS`,
  `anymesher.QuadMeshingOptions`, `anymesher.generate_hybrid_mesh_result`
- `anymesher.core.MeshCore`, `anymesher.core.corner_edges`
- `anymesher.quality_v2.triangle_quality`, `anymesher.quality_v2.quad_quality`,
  `anymesher.quality_v2.quad_candidate_quality`
- `anymesher.quad.public_integration.QuadPublicUnsupported`,
  `anymesher.quad.public_integration.QuadCapabilityMissing`,
  `anymesher.quad.public_integration.coerce_public_quad_options`,
  `anymesher.quad.public_integration.route_quad_first`
- `anymesher.surface_mesh.insert_midside_nodes`

The test imports each by its documented path and asserts the object is present
and, where it is a callable, that `callable(...)` is true. This pins the exact
re-use surface so a later tranche cannot fork to a private name.

## Test 6 — `test_quad_first_public_route_accepts_planar_orders_rejects_nonplanar`

CH2 explicitly re-opens the planar quadratic portion of the CH0 guard. `coerce_public_quad_options` accepts both `order="linear"` and `order="quadratic"` when `planar=True`, returns the same explicit `QuadMeshingOptions` instance, rejects `planar=False` for either order, and rejects unsupported order tokens such as `cubic`. Curved-surface quad-first is therefore still fail-closed while planar Q8/T6 is now a qualified public route.
## Test 7 — `test_linear_planar_route_is_the_qualified_public_contract`

The qualified public baseline remains a **real** explicit linear quad-first
mesh. The test creates one 2 m × 1 m planar source face and runs
`generate_hybrid_mesh_result` at `target_size=0.5` with
`QuadMeshingOptions(max_local_optimizations=0)`.

It asserts that the source model/revision/topology is unchanged, the published
route is `quad-first`, Q4 elements are present, Q4 diagnostics are truthful,
and Q5 is explicitly `DISABLED`. The same test also freezes
`route_quad_first(None) == (None, None)`, explicit dispatch with
`require_workers=False`, and `ELEMENT_ORDERS == ('linear', 'quadratic')`.

## Test 8 — `test_source_geometry_is_unchanged_by_promotion`

Promotion is geometry-only in a stricter sense: it **adds** midside nodes and
rewrites element node-width, but it must not move or delete any source node.
The test builds a linear quad with finite coordinates, promotes it, and asserts:

- every original source node coordinate is present, unchanged, in the promoted
  container (same values, not renumbered),
- the promoted container's node count is exactly `original + (number of distinct
  corner edges)`, and no midside node duplicates a source node location by
  coincidence of id,
- the promoted quad preserves the original four corner ids in the same order as
  the first four columns.

This mirrors the PQ invariant "source geometry identity, revision, vertices,
edges and face topology are unchanged by meshing" and asserts it holds again
once higher-order nodes are added.

## Test 9 — `test_q9_and_higher_orders_are_deferred`

Q9 (and any order beyond Q8 serendipity) is **deferred**, not absent-by-luck. The
test asserts the frozen scope boundary:

- `ELEMENT_ORDERS` is exactly `('linear', 'quadratic')` — there is no public
  `cubic`, `order-9`, or `curved` token,
- there is no `quad_options` field or top-level export enabling a ninth node,
- `docs/CURVED_QUAD_DESIGN.md` exists and names Q9 / higher orders as deferred
  (a stable substring such as `"Q9"` and `"deferred"` / `"Deferred"` appears).

This pins the scope so a later tranche that opens a higher order must explicitly
introduce a new `QuadMeshingOptions` schema version and a new shape-function
reference rather than widening the frozen Q8 contract.

---

## Gate checklist (all must pass at CH0)

| Gate | Command | Expectation |
| --- | --- | --- |
| G1 | `pytest tests/quad_first_curved/test_ch0_baseline_contract.py -q` | **9 passed**, no skip, no xfail, no worker dependency |
| G2 | `pytest tests/quad_first_planar/ -q` | No CH0-attributable regression; worker discovery uses locally built ignored outputs in this worktree |
| G3 | `pytest tests/quad_first/ -q` | All existing `quad_first` tests still pass |
| G4 | `git diff --check` | No whitespace errors in committed files |
| G5 | `git status --porcelain` (after commit) | No uncommitted changes |

No gate requires a compiled native extension. The Q4/Q5 worker executables used by broader regression gates are built locally in this worktree from the committed checkout-independent scripts; no primary-checkout worker path or environment override is part of CH0 evidence.

---

## Explicitly *not* tested at CH0 (frozen scope boundary)

- No test calls `generate_hybrid_mesh_result(..., order="quadratic")` or a curved/non-planar explicit quad-first build — those remain `QuadPublicUnsupported`. Test 7 does execute the real **linear planar** baseline route.
- No test asserts the existence of `HigherOrderEdgeRegistry`, `EnrichmentPlan`,
  or `HighOrderGeometryReport` — these are **names frozen in the design doc
  only**; they do not exist as classes yet and are not imported.
- No test evaluates explicit quad-first on a curved face (cylinder, cone, etc.); cylinder work begins in CH3 after CH1 validity and CH2 planar Q8/T6 qualification.

---

## CH1 acceptance — shared interpolation and strict higher-order validity

CH1 adds only the internal `anymesher.quad.high_order` kernel.  Public
quad-first routing remains planar/linear and `quality_v2` remains
skeleton-only.  Acceptance requires all of the following:

- Q4/Q8/T3/T6 shape values are a partition of unity, analytic gradients sum to
  zero and match independent finite differences, with the frozen corner/midside
  ordering.
- Affine mappings reproduce exactly; rigid transforms and uniform positive
  scale preserve validity and normalized mapping quality, with physical area
  scaling by the square of scale.
- Strict validity returns only `CERTIFIED_POSITIVE`, `INVALID`, or `UNRESOLVED`.
  Sampling alone never certifies.  Bernstein/subdivision bounds provide the
  proof; invalidity requires a directly evaluated non-positive witness.
- Genuinely curved positive Q8 and T6 examples certify, and dense independent
  signed-Jacobian samples lie inside the reported conservative whole-element
  lower/upper envelope.
- A shallow-budget ambiguous Q8 is `UNRESOLVED` and becomes
  `CERTIFIED_POSITIVE` with the qualified bounded budget; a known inverted Q8
  stays `INVALID` when the budget is increased.
- Hidden Q8 and T6 midside inversions are detected even though the frozen
  corner-only `quality_v2` metrics do not change.
- Reports are deterministic and JSON-safe through `to_dict()`; cancellation is
  checked deterministically and leaves inputs unchanged.

---

## CH2 acceptance — planar explicit Q8/T6 promotion

CH2 acceptance is implemented in `tests/quad_first_curved/test_ch2_planar_quadratic_promotion.py` and re-opens only the planar higher-order scope. The hard product contracts are:

- P01 h=0.5 retains all 273 linear node IDs and coordinates, all 240 shell IDs/corner connectivities and face ownership, and promotes to exactly 785 nodes / 240 Q8 / 0 T6 with 512 unique midsides. Every expanded source-edge chain contains the original linear chain at even positions.
- A real residual trapezoid retains 26 Q4 / 6 T3 as 26 Q8 / 6 T6. Shared Q8/Q8 and Q8/T6 interfaces use one midside ID, and every final high-order shell is `CERTIFIED_POSITIVE` under the CH1 validity kernel.
- P03 analytic circular-hole boundary midsides are sampled on their owning source edge at the exact midpoint parameter. The added hole midsides remain at radius 0.9 about `(3.2, 2.4)` within `1e-10`; source geometry and boundary ownership are unchanged. Bounded repair may move only unprotected opposite T6 corners and must leave all incident high-order elements certified.
- P05 concave and P07 locally graded routes retain the qualified linear corner topology/counts and pass strict high-order validity; local grading is not replaced by a global finest-size mesh.
- Promotion is idempotent. A cancellation at the final pre-publish checkpoint leaves a supplied linear mesh byte-for-byte equivalent through serialization. Source geometry is unchanged on cancellation.
- Explicit quadratic quad-first with beams/couplings fails closed rather than publishing B2 beams under `mesh.order="quadratic"`.
- Explicit curved-surface quad-first remains typed unsupported through CH2. Q9 remains deferred. `quality_v2` is unchanged.

### CH2 gate evidence

```text
python -m pytest tests/quad_first_curved/test_ch2_planar_quadratic_promotion.py -q
  -> 15 passed in 16.39 s

python -m pytest tests/quad_first_curved/test_ch0_baseline_contract.py \
  tests/quad_first_curved/test_ch1_high_order_validity.py -q
  -> 38 passed in 0.47 s

python -m pytest \
  tests/quad_first_planar/test_pq3_public_driver.py \
  tests/quad_first_planar/test_pq4_general_domains.py \
  tests/quad_first_planar/test_pq4b_staged_domains.py \
  tests/quad_first_planar/test_pq5_real_optimization.py \
  tests/quad_first_planar/test_pq6_geometry_mcf.py \
  tests/quad_first/test_q6_public_integration.py \
  tests/quad_first/test_q7_qualification.py -q
  -> 84 passed, 3 skipped in 32.00 s
     skips: compiled triangulation parity; native extension not rebuilt in this worktree

python -m pytest tests/quad_first_curved -q
  -> 53 passed in 16.39 s
python -m pytest tests/quad_first_planar -q
  -> 63 passed, 3 skipped in 39.21 s
python -m pytest tests/quad_first -q
  -> 275 passed in 3.69 s
python -m pytest tests/test_quality_and_serialize.py tests/test_coupling.py \
  tests/test_cylindrical_quadratic_staging.py -q
  -> 35 passed in 0.26 s
```
## CH3 acceptance — owner-qualified cylindrical linear quad-first

CH3 acceptance is implemented in `tests/quad_first_curved/test_ch3_cylindrical_public.py` and opens only the linear cylindrical public route. The hard contracts are:

- A source-authored pi/4 cylindrical sector can be captured as a `CylindricalQuadDomain` only through an owner-qualified binding; chart projection/lift round-trips source vertices and leaves the source model unchanged.
- Public explicit quad-first at target sizes 0.5 and 0.25 produces real target-size refinement while all shell nodes remain on the unit cylinder within `1e-10`. The coarse case is 17 nodes / 8 Q4 / 2 T3 (`N_eq=9`); the fine case is 43 nodes / 30 Q4 / 2 T3 (`N_eq=31`).
- Complete source-edge chains are published and the full authored eight-sector ring reuses the exact source-station node IDs across the periodic physical seam. The ring is 88 global nodes / 64 Q4 / 16 T3 at target size 0.5 and repeats deterministically.
- `elements_of_face` contains every final Q4 and residual T3 for each selected cylindrical face. Diagnostics report `quad-first-cylindrical`, the per-face geometry family, and genuine front-driver counters.
- Cylindrical `order="quadratic"` fails closed with `QuadPublicUnsupported` through CH3. Other unqualified curved surfaces also remain typed unsupported; the planar route and `quad_options=None` sentinel are unchanged.
- Cancellation during cylindrical face seeding propagates and the supplied GeometryModel remains unchanged; no partially published result escapes.

Qualification additionally reruns CH0/CH1/CH2, the existing cylindrical metric/atlas/patch/frontal contracts, the accepted planar PQ public regressions, and the full curved, planar, and quad-first suites. CH4 may re-open cylindrical Q8/T6 only after this CH3 linear topology handoff is committed and clean.

## CH4 acceptance — owner-qualified cylindrical Q8/T6 promotion

CH4 acceptance is implemented in `tests/quad_first_curved/test_ch4_cylindrical_quadratic.py`. It re-opens only quadratic promotion on the already-qualified CH3 cylindrical linear topology:

- h=0.5 on the authored pi/4 sector preserves the CH3 `8 Q4 / 2 T3` corner topology and shell IDs as `8 Q8 / 2 T6`, retains every linear node coordinate exactly, adds 26 unique midsides, expands every source-edge chain with the linear chain at even positions, and leaves the source model unchanged.
- h=0.25 preserves `30 Q4 / 2 T3` as `30 Q8 / 2 T6` and adds 74 unique midsides, proving the higher-order route follows genuine CH3 target-size topology rather than a fixed quadratic template.
- The full eight-sector h=0.5 ring preserves `64 Q4 / 16 T3` as `64 Q8 / 16 T6`, grows from 88 to 256 nodes, and uses one expanded quadratic station chain on the periodic physical seam. Added cylindrical midsides lie on radius 1 within `1e-10`.
- Every final Q8/T6 passes CH1 `CERTIFIED_POSITIVE` mapping validity. Shared shell edges reuse exactly one midside ID.
- Cancellation at the final quadratic-promotion checkpoint is atomic for the source model. General curved faces and Q9+ remain outside the qualified scope.

### CH4 gate evidence

- CH0+CH1+CH2+CH3+CH4 focused contract: `62 passed`.
- Cylindrical metric/atlas/patch/frontal/surface-metric/quadratic consumers plus boundary preparation: `76 passed`.
- PQ3/PQ4a/PQ4b/PQ5/PQ6 + Q6/Q7 public regressions: `84 passed, 3 skipped` (known compiled-triangulation extension capability skips).
- Full `tests/quad_first_curved`: `62 passed`.
- Full `tests/quad_first_planar`: `63 passed, 3 skipped`.
- Full `tests/quad_first`: `275 passed`.

A fresh review must additionally keep `git diff --check` clean, verify no transient CH4 evidence files are staged, and confirm the production diff is limited to cylindrical midside geometry/public scope plus the CH4 contract tests and documentation.
- Serialization / quality / coupling / quadratic-staging consumers: `35 passed`.

## CH5 acceptance — high-order geometry provenance certificate

CH5 acceptance is implemented in `tests/quad_first_curved/test_ch5_high_order_geometry_report.py`. It adds reporting only; CH2/CH4 quadratic topology and CH1 strict validity remain authoritative. The hard contracts are:

- Linear explicit quad-first reports high-order geometry as `NOT_APPLICABLE`. A successful quadratic result reports `CERTIFIED_POSITIVE` only after every final Q8/T6 has passed the CH1 strict validity gate.
- `HighOrderBoundaryMidside`, `HighOrderGeometryReport`, and `HighOrderMeshCertificate` are immutable and JSON-safe. Each boundary/interface record carries canonical source edge/station identity, exact midpoint parameter/node ID, residual and one of `straight`, `analytic_curved`, or conservative `sampled` curvature provenance.
- Each face report includes source model/revision, geometry family, chart kind and lifted 3D chart origin, Q8/T6 and certification counts, boundary/interior projection methods, curvature classes and geometry residuals. Positive face reports must certify every owned element.
- The global certificate rejects duplicate face reports, model/revision mismatch, Q8/T6 count mismatch, conflicting canonical boundary provenance, inconsistent unique-boundary counts, or a residual envelope inconsistent with its face reports.
- P01, P03, one owner-qualified cylindrical sector and the full eight-sector ring reproduce the CH2/CH4 topology/counts exactly while adding truthful provenance. P03 includes straight and analytic curved source intervals; the full ring has 72 globally unique canonical source-boundary/interface midsides with no conflicting payload.
- P01 maximum geometry residual is `0.0`; P03 is `5.140558480280521e-13`; the cylinder sector and full ring are `2.854117595420714e-13` and `2.8563285926006375e-13`. All are below the qualified `1e-10` owner tolerance.
- Cancellation at the final quadratic publication checkpoint raises before either topology or certificate publication; source geometry remains unchanged.

### CH5 gate evidence

- Focused CH5 contract: `8 passed`.
- Full `tests/quad_first_curved`: `70 passed` after fresh-review strengthening of chart-origin, curvature-class and global-consistency contracts.
- Full `tests/quad_first_planar`: `63 passed, 3 skipped`; the skips remain the known compiled-triangulation capability skips because the native extension is not rebuilt in this worktree.
- Full `tests/quad_first`: `275 passed`.
- The pre-review cylindrical consumer bundle remained green at `69 passed`; the post-review changes are confined to the new report metadata/consistency path and the explicit quad-first quadratic report assembly.

A fresh review must keep `git diff --check` clean, remove all transient `.ch5_*`/gate evidence helpers before commit, and verify that CH5 does not open cones, ruled/Coons surfaces, Q9, a second topology engine, or a new public option schema.


## CH6 acceptance — coherent B3 on quadratic quad-first mixed output

CH6 acceptance is implemented by `tests/quad_first_curved/test_ch6_quadratic_beam_ownership.py`. A straight member-through-planar-face fixture that was deliberately fail-closed at CH2 must now complete through the public explicit quadratic quad-first route.

Acceptance requires all final shell connectivities to be Q8/T6, every beam connectivity to contain exactly three nodes in start-mid-end order, and every B3 middle node to equal the endpoint chord midpoint within `1e-12`. The member `nodes_of_edge` sequence must contain one quadratic midside between each retained linear station, and the through-face structural coupling must reference the exact member station at the plate intersection.

The accepted target-size-1 fixture publishes `13 nodes / 1 Q8 / 0 T6 / 2 B3 / 1 coupling`; the member edge chain has five stations and the coupling uses its centre station. Repeated generation is deterministic and source geometry is unchanged. Cancellation at the quadratic publication checkpoint is atomic.

Curved quadratic beam edges remain rejected by the existing straight-sided B3 rule. Cylindrical quadratic shell/beam requests remain typed unsupported in CH6. The CH2 direct-promotion test for arbitrary supplied B2 beam content remains fail-closed; CH6 only qualifies beams produced coherently by the public mixed generation route.

## CH7 acceptance — cylindrical quadratic B3 ownership

CH7 acceptance is implemented in `tests/quad_first_curved/test_ch7_cylindrical_b3_ownership.py`. It qualifies only straight B3/member content on the already-qualified CH4 cylindrical Q8/T6 route.

The full authored eight-sector ring at target size 0.5 with one radial member must publish exactly `261 nodes / 64 Q8 / 16 T6 / 2 B3 / 1 coupling`. Every beam body has start-mid-end width three and each B3 middle node is the endpoint chord midpoint within `1e-12`. The member station at radius 1 is the exact owner-cylinder attachment point; the shell remains on radius 1 within `1e-10`.

Attachment resolution may widen tolerance only for quadratic target-face elements and only by the mesh-derived midside/chord deviation. The accepted coupling records a nonzero but bounded eccentricity below `1e-3`; no coordinate welding or owner-point movement is permitted. Serialization must preserve B3 connectivity, member station chains and coupling ownership.

Repeated public generation is deterministic, source geometry is unchanged, and cancellation at `quad-first:quadratic-promotion-ready` is atomic. Curved B3 lines remain typed unsupported through the existing CH6 straight-sided preflight.

Qualification evidence: focused CH6+CH7 plus structural/coupling/serialization consumers `63 passed`; full `tests/quad_first_curved` `75 passed`; full `tests/quad_first_planar` `63 passed, 3 skipped` for the known unreconstructed native triangulation capability; full `tests/quad_first` `275 passed`.

## CH8 acceptance — analytic conical linear quad-first

CH8 acceptance is implemented in `tests/quad_first_curved/test_ch8_conical_public.py` and opens only the linear analytic Cone route.

- `ConicalQuadDomain` must bind model/revision/face identity, round-trip owner positions through the analytic developed chart, and leave source geometry unchanged.
- Public h=0.6 and h=0.3 executions must show real target-size causality, positive/full chart-space coverage, complete intrinsic source-edge chains, all final Q4 plus residual T3 in `elements_of_face`, owner support residual <= `1e-10`, and genuine front counters under route `quad-first-conical`.
- Repeat generation must be deterministic. Cancellation during face seeding or before publication must propagate without partial output or source mutation.
- Conical `order="quadratic"` remains typed unsupported through CH8. Existing planar and cylindrical public routes remain unchanged; ruled/Coons/general curved surfaces and Q9 are not opened.

### CH8 gate evidence

- focused CH8 product: `7 passed`;
- CH0 through CH7 focused curved contracts: `75 passed in 52.04s`;
- curved native qualification plus chart/physical-quality/serialization consumers: `43 passed in 8.58s`;
- accepted PQ3/PQ4a/PQ4b/PQ5/PQ6 + Q6/Q7 public regression bundle: `84 passed, 3 skipped in 31.76s`; the three skips are the known unreconstructed native triangulation parity capability checks;
- full `tests/quad_first_curved`: `82 passed in 53.83s`;
- full `tests/quad_first_planar`: `63 passed, 3 skipped in 38.44s`;
- full `tests/quad_first`: `275 passed in 3.67s`.

Fresh review additionally verifies the developed-map inverse branch, local metric isometry, source/revision binding, exact edge ownership, target-size topology change, cancellation, and that no conical quadratic, ruled/Coons, or Q9 route was activated.

## CH9 acceptance — analytic conical Q8/T6 promotion

CH9 acceptance is implemented in `tests/quad_first_curved/test_ch9_conical_quadratic.py` and re-opens only quadratic promotion on the accepted CH8 analytic Cone topology.

- h=0.6 preserves CH8 `28 Q4 / 6 T3` shell IDs and corner topology as `28 Q8 / 6 T6`, retaining all 42 linear node IDs/coordinates and growing to exactly 117 nodes with 75 unique midsides.
- h=0.3 preserves `89 Q4 / 14 T3` as `89 Q8 / 14 T6`, retaining all 118 linear nodes and growing to exactly 338 nodes with 220 unique midsides.
- Every shared final shell edge has exactly one midside ID. Every source-edge chain expands with the CH8 linear chain at even positions.
- Interior conical midsides are developed-chart midpoint lifts, not raw 3D chord midpoints. Final owner support residual is <= `1e-10` and every Q8/T6 passes CH1 `CERTIFIED_POSITIVE` validity.
- `high_order_geometry` reports `geometry_family="conical"`, `chart_kind="ConicalQuadDomain"`, `interior_projection="owner-chart-midpoint"`, correct Q8/T6/unique-midside counts, and a bounded owner residual envelope.
- Repeat generation is deterministic. Cancellation at `quad-first:quadratic-promotion-ready` is atomic and leaves source geometry unchanged.
- Conical quadratic requests with beam/coupling content remain typed unsupported. Planar/cylindrical behavior is unchanged; ruled/Coons surfaces and Q9+ remain outside CH9.

### CH9 gate evidence

- focused CH9 product: `5 passed in 6.38s`;
- full `tests/quad_first_curved`: `87 passed in 78.19s`;
- full `tests/quad_first_planar`: `63 passed, 3 skipped in 40.75s` (known unreconstructed native triangulation parity capability skips);
- full `tests/quad_first`: `275 passed in 5.60s`;
- curved-native / quality / coupling / quadratic-staging consumers: `41 passed in 8.19s`;
- `git diff --check`: clean before documentation closeout.

Fresh review must confirm no topology-ID drift, exact owner-chart interior midsides, exact source-edge boundary ownership, strict CH1 validity, CH5 provenance truth, cancellation atomicity, and no accidental conical B3, ruled/Coons or Q9 activation.


## CH10 acceptance — metric ruled/Coons linear quad-first

CH10 acceptance is implemented in `tests/quad_first_curved/test_ch10_parametric_curved_linear.py` and opens only explicit linear quad-first meshing for non-planar `RuledSurface` and `CoonsSurface` faces.

- `ParametricQuadDomain` must bind model/revision/face identity, round-trip owner positions through `FaceChart`, expose positive chart area and leave source geometry unchanged.
- Public h=0.6 and h=0.3 executions must show real physical target-size causality: increasing `N_eq`, decreasing maximum physical shell corner edge, complete source-edge chains, all final elements in `elements_of_face`, real front counters and owner support residual <= `1e-10`.
- Repeat generation must be deterministic. Cancellation during face seeding must propagate without source mutation or partial publication.
- Diagnostics must report route `quad-first-parametric-curved` and the exact per-face family (`ruled` or `coons`).
- Ruled/Coons `order="quadratic"` remains typed unsupported through CH10. Existing Plane, Cylinder and Cone routes remain unchanged; Q9+ is not opened.

### CH10 gate evidence

- focused CH10 product: `8 passed in 0.74s`;
- chart / physical-quality / serialization / coupling consumers: `47 passed in 0.73s`;
- accepted PQ3/PQ4a/PQ4b/PQ5/PQ6 + Q6/Q7 public regression bundle: `84 passed, 3 skipped in 31.60s`; skips remain the known unreconstructed native triangulation parity capability checks;
- full `tests/quad_first_curved`: `95 passed in 60.37s`;
- full `tests/quad_first_planar`: `63 passed, 3 skipped in 38.56s`;
- full `tests/quad_first`: `275 passed in 5.51s`.

Fresh review must keep the centre-metric normalization truthful as an approximate physical spacing map rather than an exact global isometry, preserve exact geometry-owned boundary station identity, source immutability and cancellation atomicity, retain the single accepted front/driver topology pipeline, and confirm no ruled/Coons quadratic route is activated before CH11.


## CH11 acceptance — ruled/Coons Q8/T6 promotion

CH11 acceptance is implemented in `tests/quad_first_curved/test_ch11_parametric_curved_quadratic.py` and re-opens only quadratic promotion on the accepted CH10 non-planar ruled/Coons topology.

- h=0.6 must preserve `8 Q4 / 0 T3` as `8 Q8 / 0 T6`, retaining all 15 linear nodes and growing to exactly 37 nodes with 22 unique midsides for both ruled and Coons fixtures.
- h=0.3 must preserve `28 Q4 / 0 T3` as `28 Q8 / 0 T6`, retaining all 40 linear nodes and growing to exactly 107 nodes with 67 unique midsides.
- A real locally graded residual fixture must preserve `25 Q4 / 4 T3` as `25 Q8 / 4 T6`, growing from 35 to 98 nodes with 63 unique midsides and no topology drift.
- Every shared final shell edge must have one midside. Every source-edge chain must contain the CH10 linear chain at even positions. Interior midsides must be owner-chart midpoint lifts through `ParametricQuadDomain`, while boundary midsides retain exact source-edge parameter ownership.
- Every Q8/T6 must pass CH1 `CERTIFIED_POSITIVE`; the CH5 certificate must report the exact `ruled`/`coons` family, `ParametricQuadDomain` chart kind and `owner-chart-midpoint` interior projection with owner residual <= `1e-10`.
- Repeat generation must be deterministic and cancellation at `quad-first:quadratic-promotion-ready` atomic. Quadratic ruled/Coons with beam/coupling content remains typed unsupported.

### CH11 gate evidence

- focused CH11 product including the graded Q8/T6 fixture: `11 passed in 5.52s`;
- full `tests/quad_first_curved` after CH11 and the evolved CH10 linear contract: `106 passed in 65.21s`;
- full `tests/quad_first_planar`: `63 passed, 3 skipped in 39.04s`; skips remain the known unreconstructed native triangulation parity capability checks;
- full `tests/quad_first`: `275 passed in 5.67s`;
- chart / physical-quality / serialization / coupling / quadratic-staging consumers: `58 passed in 0.73s`.

Fresh review must confirm no corner/topology-ID drift, owner-chart rather than chord interior midsides, exact source-edge boundary ownership, strict CH1 validity, truthful CH5 provenance, cancellation atomicity, and no accidental curved B3 or Q9 activation.
