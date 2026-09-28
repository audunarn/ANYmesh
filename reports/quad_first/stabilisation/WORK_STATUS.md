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
