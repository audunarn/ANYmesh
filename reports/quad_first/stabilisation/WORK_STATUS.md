# Quad-first stabilisation work status

Branch: `claude/anymesher-050-release-4xztmm`, based on `main` at `2a52a7b`.
Scope (approved plan): phase 1 green CI and experimental docs, phase 2 shape
gates, phase 3 performance. Transfinite four-sided layouts, global blossom
matching with parity, and the smoothing rework are deferred.

Context: ANYmesher 0.5.0 on PyPI was built from `2ccef37` and does not contain
the quad-first route. The route is documented as an experimental opt-in under
*Unreleased*.

## Phase 1: green source matrix, worker build, docs

- `tools/build_quad_workers.py` builds both optional workers on Windows
  (`cl` in a developer shell), Linux and macOS (`$CXX`/`c++`/`g++`/`clang++`)
  with `-DEIGEN_MPL2_ONLY`. The TinyAD worker's platform headers were moved
  out of its anonymous namespace so it compiles with libstdc++.
- One output naming rule (`.exe` on Windows, no suffix elsewhere); the legacy
  POSIX `quad_mcf_worker.out` name is still accepted.
- `quad_workers` pytest marker: explicit skip when absent, failure under
  `ANYMESHER_REQUIRE_QUAD_WORKERS=1`. CI builds and requires the workers.
- Evidence (Linux, Python 3.11, ANYgeometry `7edbb8b`):
  - workers absent: quad suites `295 passed, 46 skipped` (worker skips explicit);
  - workers present and required: `338 passed, 3 skipped` (native parity only).

Two curved tests were already red on `main` against the CI-pinned ANYgeometry,
hidden by the IS1 collection error. The open two-face cylinder test now
asserts the explicit refusal when the owner API is absent, matching the IS1
curved test. The CH9 fine-cone count is updated with the phase 2 values below.

## Phase 2: shape gates

Before phase 2, the route admitted any strictly convex quad under an absolute
`1e-9` tolerance. Independent measurement on seven small planar shapes found
a 179.9999996-degree Q4 corner, a Q4 aspect ratio of 352 and residual T3
corners down to 0.6 degrees. The CH8 cone at h=0.6 had a 177.3-degree corner
and a 2.7-degree triangle.

Changes:

- `quad/quality_gate.py` holds the shape policy, built on `structured.MeshQualityPolicy`:
  - Q4: scaled Jacobian at least 0.20, corners within [20, 160] degrees, aspect ratio at most 10;
  - residual T3: minimum corner angle 15 degrees.
- The Q4 gate is enforced in `front.make_quad`, which every Q4 creation path
  uses, with scale-free degeneracy tolerances. The residual T3 gate is
  enforced for recovery split children.
- `quad/residual.py` adds a bounded, transactional pass of residual T3/T3
  diagonal flips that raise the pair's minimum angle; it never touches
  protected edges.
- Lattice seeds skip points within 0.3 h of a boundary segment. A 0.45 h
  clearance was tried and rejected because it degraded the ruled/Coons grids.
- The boundary registry no longer applies mapped opposite-side equality on
  edges owned only by quad-first faces whose declared mapped side spans
  several edges. That equality put 31 divisions on a 1.2 m edge of an
  L-shaped plate at h=0.2.
- The final validator raises a typed `QuadQualityRejected` and reports
  worst-element metrics plus the active gates. The IS1 repair fails closed
  instead of publishing `unresolved_quads`.

Independent audit (published mesh, quality_v2 definitions):

| shape (h) | before: Q4/T3, worst Q4 angle, min SJ, thinnest T3 | after |
| --- | --- | --- |
| triangle (0.35) | 131/18, 180.0, 0.000, 9.1 | 126/18, 156.4, 0.401, 17.6 |
| pentagon (0.35) | 234/18, 173.1, 0.120, 10.1 | 220/21, 159.6, 0.348, 21.0 |
| parallelogram (0.4) | 136/16, 172.9, 0.123, 2.2 | 140/8, 158.6, 0.364, 30.0 |
| trapezoid (0.25) | 204/14, 168.7, 0.196, 11.3 | 197/16, 159.0, 0.359, 21.0 |
| plate with hole (0.4) | 356/12, 153.6, 0.445, 45.0 | 354/12, 157.6, 0.382, 37.9 |
| L-shape (0.2) | 197/36, 142.2, 0.613, 9.2 | 197/0, 90.0, 1.000, n/a |

All adaptive-layout runs of the same shapes also pass the gates, as pinned by
`tests/quad_first_planar/test_pq7_quality_gates.py`.

Frozen evidence updated because of the gates (old -> new):

- CH8/CH9 cone h=0.6 linear: 42 nodes / 28 Q4 / 6 T3 -> 37 / 23 / 6.
  Quadratic: 117 -> 102 nodes, 75 -> 65 midsides. The old mesh contained a
  177.3-degree Q4 and a 2.7-degree T3.
- CH9 cone h=0.3: 118 / 89 / 14 -> 119 / 91 / 12; quadratic 338 -> 340 nodes,
  220 -> 221 midsides. `main` already gave 122 / 93 / 14 against the pinned
  geometry.
- CH12 cone plus member: 28 -> 23 Q8.
- CH4 cylinder sector h=0.25: 30 Q4 / 2 T3 -> 24 Q4 / 0 T3 (all-quad).
- CH2 P03 hole: the quadratic T6 corner repair is no longer needed (3 -> 0),
  so that repair path currently has no public-fixture coverage.
- PQ6 skew corridor: 78 -> 76 seed T3. MCF now solves two 2-cell components
  (2 calls, 2 pairs; was one 6-cell component, 3 pairs). Final result
  38 Q4 / 2 T3 -> 37 / 2. MCF still reduces front attempts (41 vs 43).
- PQ3 trapezoid: recovery is still causal (27 vs 23 Q4, 4 T3 both). The
  assertion moved from "fewer T3" to "more Q4".
- Q2 conforming-recovery fixtures now find their Q4/T3 interface edge
  topologically instead of by hard-coded node ids.

Evidence: quad-first planar, curved and core suites with workers required:
`486 passed, 3 skipped`; seeding, triangulation contract, quality/serialize,
coupling and layering regressions: `49 passed, 4 skipped`.

## Phase 3: performance (results unchanged)

- `QuadMeshState` keeps per-kind cell counters updated on commit
  (`count_kind`). The front driver no longer scans every cell on each
  iteration to test for residual T3; the MCF stage uses the same counter.
- Shared Python Bowyer-Watson (`triangulation._bowyer_watson`):
  - A conservative circumcircle pre-filter skips triangles the new point is
    provably outside. The bound uses the propagated circumcentre error, so
    needle triangles against the super-triangle stay candidates.
  - Every remaining candidate is still decided by the adaptive `incircle`.
  - The per-insertion list sort, which never affected the result, was dropped.
  - Results equal the original on 640 randomized and adversarial point sets:
    cocircular lattices and rings, 1e-9 to 1e7 scales, collinear runs,
    jittered lattices. The original is kept as an oracle in
    `tests/test_triangulation_prefilter.py`.
- `triangulation._point_in_ring` and the PSLG on-ring checks are vectorized.
  They use the same bounding-box comparisons in front of the exact
  `_point_on_segment`, and the same crossing arithmetic. They matched the
  scalar loop on 75,798 queries (also pinned by a test).
- Seeding spacing and boundary-clearance checks use uniform-grid buckets.
  Interior seeds are byte-identical to the full scans on 28
  uniform/graded/adaptive cases.
- `_SegmentGrid.closer_than` (phase 2) now scans every bucket covering the
  query square. Before, it could miss a segment when the query radius
  exceeded the cell size on graded fine points. That changes the CH11 graded
  ruled/Coons fixture from 35 nodes / 25 Q4 / 4 T3 to 34 / 24 / 4
  (quadratic 98 -> 95 nodes, 63 -> 61 midsides).
- Seeds stay on the Python reference triangulation. The compiled triangulator
  breaks exactly cocircular ties (every rectangular lattice cell) differently.
  Its seeds are valid but not byte-identical outside the planar PQ4b parity
  corpus: the CH9 h=0.3 cone gives 115 / 86 / 14 natively versus 119 / 91 / 12
  in Python. Using it would make quad-first output depend on how ANYmesher
  was installed. This is a native/Python parity gap to address separately.

`benchmarks/quad_first_scaling.py`, 10 m x 6 m plate, existing layout, one
unrepeated run per size in this container:

| Q4 | before, phase-1 commit (s) | after (s) |
| ---: | ---: | ---: |
| 240 | 0.75 | 0.39 |
| 960 | 8.81 | 2.10 |
| 3,094 | 72.06 | 12.69 |
| 6,000 | not finished (~280 extrapolated) | 30.97 |

The "before" runs had no workers built, so their Q4/Q5 stages were skipped;
the "after" runs include the worker calls. The difference therefore
understates the gain.

The empirical exponent is now about 1.2 to 1.5. Seeding is still the largest
stage (21.6 s of 31.0 s at 6,000 Q4): per-segment `_edge_incidence` rebuilds
and PSLG classification remain O(N) per call. Q5 candidate ranking is 7.3 s.

## CI follow-up (run 94)

Run 94 on `041ccaa` passed all 4 Windows and all 4 Linux pytest cells, the
wheel builds, Gmsh, native-v2 contract and native-absent jobs. The worker
build step passed on all 12 pytest cells. Two failures remained:

- **Installed-wheel full suite:** collection stopped at the IS1 test because
  the isolated job (`python -I`, `-o pythonpath=`) had no repository root on
  `sys.path`, so `benchmarks.is1` and `benchmarks.sg1` could not be imported.
  `tests/conftest.py` now appends the root at the lowest priority. The root
  has no `anymesher` package, so the installed distribution is still the one
  tested. Reproduced and verified locally with the job's isolated invocation.
  `MANIFEST.in` also ships `benchmarks/is1` and `benchmarks/sg1`, so the
  sdist's own tests collect.
- **macOS (4 cells), CH9 h=0.3 cone:** macOS gives 117 / 89 / 12 where Linux
  and Windows give 119 / 91 / 12. Diagnosis:
  - No shape gate, clearance, MCF or Q5 decision is near its threshold; the
    smallest gate margin is 0.15%.
  - One-ulp noise in the chart projection or in edge sampling does not change
    the result.
  - One-ulp noise in `math.hypot` inside the cross-field guidance does: 5 of 6
    perturbed runs give 118 / 90 / 12.
  - `guidance.rank_bodies` sorts candidate quads by float score, and
    theoretically tied candidates are separated by last-bit libm
    differences.
  - The test now asserts platform-independent invariants instead: exact
    linear-to-quadratic topology, midsides equal to unique linear edges,
    quadratic nodes equal to linear nodes plus midsides, and a narrow
    quad-dominant count band.
  - The mesher is unchanged. Follow-up option: quantise the ranking score
    before the deterministic body tie-break, so exact-in-theory ties no
    longer depend on the platform's libm.

## Phase 4: remaining whole-mesh scans (results unchanged)

Branch `claude/quad-first-perf-0.5.1`, based on `main` at `1baa6c5` (0.5.1).

Question: the 0.5.1 note left 6,000 Q4 at ~31 s (container) with seeding
dominant and an empirical exponent of 1.2-1.5. Is the remaining cost
superlinear scans or intrinsic work? Profiling at 6,000 and 12,298 Q4 showed
whole-mesh scans. Each one was replaced by an exact local equivalent, and the
replaced loop is kept as a test oracle:

| Stage | Scan removed | Replacement |
| --- | --- | --- |
| CDT | full `_edge_incidence` rebuild per segment, even when the segment already was an edge | one edge set, rebuilt only after an actual recovery |
| CDT | `_bowyer_watson` list and array rebuild plus an O(T) circumcircle filter per insertion | reusable slots, and a uniform grid of conservatively enlarged circumdisks; unbounded or huge disks (super-triangle fans, needles) stay on a global list |
| CDT | PSLG splitting scalar `_point_on_segment` over every point for every segment | the same bbox comparisons, vectorized, in front of the scalar test |
| CDT | ring arrays rebuilt per point query in `_prepare_pslg` and `_finish_triangles` | computed once per ring |
| Q5 | `QuadMeshState.cells_at` scanned every cell (called for every node by `_eligible`) | union of `_edge_to_cells` over `_node_edges[node]` |
| Seed | scalar `_strict_inside` per lattice candidate (uniform, graded, adaptive) | `_strict_inside_mask`, same float operations over the whole grid |

Grid soundness: a triangle is registered by the bbox of radius
`sqrt(r2)*(1+1e-6) + 4*delta + 1e-9*(|a|+|u|+span)`. That radius contains the
exact circumcircle and every point the existing pre-filter keeps (its slack is
1e-8 relative). Any triangle may be global, so the super-triangle fan is
global without that arithmetic. The cavity is still decided by the exact
`incircle`, and the insertion order is unchanged. The order matters: on
cocircular lattices it selects the triangulation. The remaining
Bowyer-Watson cost is the ~27 triangles created per insertion. They come from
the fan churn of x-sorted insertion, and changing the order would change the
output.

Evidence (Windows, Python 3.11.9, ANYgeometry 0.4.4, workers built):

- Fingerprints: SHA-256 of `mesh_to_dict` plus `hybrid_diagnostics`, with
  volatile keys removed (model UUID, timings, paths). Result identical to
  `main` on 86/86 cases. The 72 IS1 cases cover all six fixtures x two sizes
  x existing/adaptive x uniform/refined, linear plus quadratic on uniform.
  The 14 plate cases cover the benchmark plate with and without the hole,
  both layouts, up to 6,000 Q4.
- `tests/test_scan_free_equivalence.py`: `cells_at` against a full scan
  through 60 random commits (and on staged views); `_strict_inside_mask` and
  `_uniform_lattice_points` against the scalar loops; segment-recovery skip
  against recovering every segment (the cases do need recovery); grid
  Bowyer-Watson against the original full scan on six larger sets (lattice,
  jittered 1e5 lattice, random strip, clustered with outliers, concentric
  cocircular rings). Shrinking the grid disks by 50% makes 30 of 42 oracle
  cases fail, so the oracle detects an unsound grid.
- `tests/test_triangulation_prefilter.py` (121 adversarial sets) still passes.
- Full suite with `ANYMESHER_REQUIRE_QUAD_WORKERS=1` (before the final
  adaptive-seed change): 1901 passed, 29 skipped. One failure: the sdist
  packaging test, because the fresh venv lacked `build`; it passes once
  `build` is installed. Rerun on the final commit (quad-first suites plus
  every seeding, triangulation and layout test): 1093 passed, 6 skipped.

`benchmarks/quad_first_scaling.py`, one unrepeated run per size. Timings
varied by about 30% between runs on this shared machine; "before" is `main`
at `1baa6c5` on the same machine and day:

| Case | Q4 | before (s) | after (s) |
| --- | ---: | ---: | ---: |
| plate, existing | 960 | 0.87-0.91 | 0.52-0.60 |
| plate, existing | 3,094 | 4.8-5.9 | 2.5-2.6 |
| plate, existing | 6,000 | 11.9-16.3 | 3.8-5.2 |
| plate, existing | 12,298 | 51.7 | 7.7-9.9 |
| plate, adaptive | 6,157 | 19.3-19.8 | 4.7-5.2 |
| plate with hole, adaptive | 5,734 | 19.4-23.7 | 6.0-6.4 |

The empirical exponent from 6,000 to 12,298 Q4 is 0.9-1.3, down from 1.6.

Remaining limits and follow-ups:

- Small refined curved cases (for example IS1 Coons h=0.3, adaptive,
  refined: ~10 s for 184 cells) spend ~80% in `_repair_quad_first_quality`
  normal checks. That time is in ANYgeometry `closest_uv` point projection
  (~14.5k calls). ANYgeometry owns it; not changed here.
- `_recover_segment` still rebuilds incidence once per flip. That only
  matters for many long constraints that need recovery; the benchmarks have
  none.
- MCF `_pair_graph` / `make_quad` and the front step are linear, but with
  high constant factors (`_int_or_err`/`position` per access).

## Phase 5: curved paths and owner geometry (results unchanged)

Question: after phase 4, what makes small curved and refined meshes slow (for
example ~10 s for 184 cells on a refined Coons face)? The phase-4 note
attributed it to ANYgeometry projection. The user authorized changes in
ANYgeometry for this release. Profiling the whole IS1 corpus and the slowest
real-workflow tests found these costs. Every replacement is exact and keeps
the replaced code as a bit-for-bit oracle in tests.

| Owner | Cost | Replacement |
| --- | --- | --- |
| ANYmesher `charts.FaceChart` | `import_module("anygeometry.meshing")` failed and rescanned the filesystem on every chart call (~196k calls); `inspect.signature` was rebuilt per call | import resolved once per process; signatures cached per function |
| ANYmesher `high_order` | the validity Jacobian built `shape_gradients` via column stacking plus `np.cross` for every point, and the root patch twice | the same element-wise expressions into the same array layout, a multiply-then-subtract cross, and a per-certification memo (signed zeros distinct) |
| ANYgeometry surfaces | Coons/ruled `_sample` called `np.clip` on scalars; `closest_uv` ran point by point from `face_local_uv_many` and `face_trim_loops_uv` | scalar clamp (identical for signed zeros, infinities and NaN); a batched Gauss-Newton with the scalar element-wise order and per-point `lstsq` |
| ANYgeometry cylinders/cones | `circumferential_direction` recomputed `np.cross` per access | computed once as a non-field attribute; callers get a copy |
| ANYgeometry trims | `face_trim_loops_uv` re-projected every trim vertex on every `project_to_face` | reused for the same committed revision and `Face` object; never inside a transaction; cleared on clone and deserialization; copies returned |
| ANYgeometry topology Coons | 8 boundary-chain evaluations per point, 4 of them the constant corners; per-call length arrays, id validation and ~40 tiny NumPy operations | corners and resolved chain data cached under the trim-loop rule (a miss keeps the original evaluation and error order); length arrays keyed by the exact lengths; blending per component in the former order |
| ANYgeometry atlas proof | `Fraction` products and roundings; `atan_series`, `atan2`, `sincos` and `cross` recomputed repeatedly within one proof | exact integer rounding for `add`/`mul`, with the original code whenever a bit budget could bind; per-proof memo that replays the recorded work charges (π-dependent operations only after π exists). Queries and binding validation still requalify from scratch. |
| ANYgeometry trim membership | edge loop of 1-element NumPy operations per point | points × edges evaluated at once for small finite inputs |

Evidence (same environment as phase 4; ANYgeometry from
`claude/projection-perf` at `C:/Github/ANYgeometry-perf-045`, based on
`origin/main` `c0f1d80`, source-identical to 0.4.4):

- ANYmesher fingerprints: 86/86 identical to `main` with ANYgeometry 0.4.4.
- Hole-punched 2 x 2 plate (`test_operations` butterfly, h = 0.15): full
  `mesh_to_dict` identical apart from `preparation_hash` and
  `structural_preparation_hash`, which already differ between two runs of the
  same code because they hash the random model ID.
- Cylinder atlas: complete `CylinderAtlasResult` content (sectors,
  occurrences, intervals, certificates with work counts) identical to 0.4.4
  on 10 queries plus validations. In ANYgeometry tests, the memoized proof
  matches recomputation in results, work counts, callback sequences and
  budget refusals.
- ANYgeometry suite: 987 passed. 3 release-authority tests failed on a local
  `git push` to a temporary origin; the unmodified checkout fails 6 of those
  7 in this environment.
- ANYmesher full suite with the new ANYgeometry and
  `ANYMESHER_REQUIRE_QUAD_WORKERS=1`: 1925 passed, 29 skipped, in 20 min.
  The phase-4 run with 0.4.4 took 50 min.

Timings (same machine, sequential runs):

| Case | main + 0.4.4 (s) | this branch + candidate (s) |
| --- | ---: | ---: |
| IS1 corpus, 72 meshes | 104.3 | 31.9 |
| - Coons cases | 29.7 | 6.1 |
| - ruled cases | 16.4 | 5.1 |
| - cylinder cases | 40.3 | 13.9 |
| - cone cases | 14.5 | 4.6 |
| hole-punched plate, one `generate_mesh` (under load) | 646 | 165 |
| `test_punching_a_hole_leaves_a_meshable_ring` (under load) | 1302 | 437 |

Remaining limits:

- Structural-closure preparation (`find_coplanar_overlaps` -> curved
  face-face qualification) still dominates hole-punched and stripped-cylinder
  models. Its scalar `closest_uv` on topology surfaces always runs all 30
  finite-difference Gauss-Newton iterations. Reducing that changes the
  numerical path and needs a qualified ANYgeometry change, not an exact
  refactor.
- The atlas proof and IS1 repair still use per-point `np.linalg.lstsq`.
  Batching it would change the last bits.
- The ANYgeometry changes need that owner's review and release (0.4.5).
  ANYmesher does not require them.
