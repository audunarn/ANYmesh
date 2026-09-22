# Curved and Higher-Order Quad Meshing — Acceptance Record (CH0–CH1)

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

## Test 6 — `test_quad_first_public_route_rejects_nonlinear_nonplanar`

The public route stays **linear + planar only** at CH0, and the guard is not
accidentally removed. `QuadPublicUnsupported` (src/anymesher/quad/public_integration.py:34)
must be raised, from `coerce_public_quad_options`
(src/anymesher/quad/public_integration.py), for:

1. `order="quadratic", planar=True` — higher-order is not yet a public quad-first
   route.
2. `order="linear", planar=False` — curved is not yet a public quad-first route.

And for the accepted pair `order="linear", planar=True` the call must return the
same `QuadMeshingOptions` instance (no error). This is the boundary between the
qualified planar route and the curved/HO scope; both guards must fire.

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
