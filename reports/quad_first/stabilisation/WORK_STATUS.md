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
