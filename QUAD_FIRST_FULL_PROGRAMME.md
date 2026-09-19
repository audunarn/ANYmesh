# ANYmesher Native Quad-First — Full Programme Execution Brief

Technical lead: ChatGPT Astra chat mode. Implementation lead: OpenCode. Primary coding model: ollama/qwen3.8:27b. Worktree: C:\Github\ANYmesh\.worktrees\quad-first-v1. Branch: opencode/quad-first-v1. Baseline: 2ccef378c3efb4ba3a9957b9ab896ac8fc454b9d.

Authority: implement the full Q0-Q7 / M1-M3 Native Quad-First programme autonomously. Do not stop after M1. Stop only for a genuinely critical issue: destructive/irreversible action, conflict with unrelated user work, essential unavailable dependency/source, or a hard engineering invariant that cannot be preserved after bounded investigation.

Goal: deliver an opt-in planar, boundary/field-guided, quad-dominant mesher. Quad-first means quad-driven construction using a valid background CDT, not triangle-free construction and not renamed greedy recombination. Existing mapped/native defaults, old schemas and closed native-v2 performance evidence remain untouched.

Hard invariants:
- ANYgeometry remains authoritative for geometry/topology/ownership.
- Preserve exact shared node IDs, exact parameter stations, reversed shared-edge use, Sheet ownership, material/thickness partitions and component transactions.
- No coordinate welding or silent input repair.
- Existing NativeMeshingOptions schema is unchanged; add a separate QuadMeshingOptions.
- Existing _native extension stays C++17 and ABI-stable.
- Deterministic ordering/ties, cancellation, typed failure and rollback are mandatory.
- Residual triangles must be independently qualified S3/T3 closures; do not publish helper triangles underneath active quads.
- No hidden fallback counted as quad-first success.
Reuse policy:
- Selectively port/adapt MIT KarlLevik/qmorph at commit bb798752b998489497f4f38b5989c42d13ac2bc0. Preserve attribution. No production Java/runtime/viewer/file-loader/static-global-state/fake-quads/unbounded retry logic.
- Integrate MIT libSatsuma commit 4e96979ecb11bbfe8d9c05e8f8be1ecb992ca5fd for supported nontrivial bidirected integer subdivision problems. No Gurobi or Blossom V. LEMON cgg commit 3c6aa54c62a9b524bb9502872fa172776c1f8244 is Boost-1.0. libTimekeeper commit f4f486648faac7c740535678b45b387078e26918 is MIT.
- Integrate MIT TinyAD commit 4b48d1a1a588874556a692a3abbdecd0db4c23e1 for bounded local Q4 quality optimization. Use Eigen MPL2-compatible headers with EIGEN_MPL2_ONLY; resolve/pin exact Eigen source/archive hash during Q0.
- No QuadWild, VCGLib, Qt, OpenVolumeMesh, Java VM, commercial solver, GPL dependency or installation-time network fetches in production.
- Record all third-party pins, hashes, source paths, notices, patches and build settings in third_party/quad/manifest.json and third_party/quad/licenses.

Target architecture:
- Existing structural preparation -> immutable quad input contract.
- One valid constrained CDT seed, avoiding the full old quality/refinement/recombination pipeline before quad-front construction.
- Separate planar size metric and cross-field orientation guidance.
- Existing simple seeding where sufficient; libSatsuma only for representable nontrivial internal count systems.
- Native mixed working state: residual T3 + accepted Q4, local edge incidence, front halfedges, feature flags, stable/generation handles, transaction journal.
- Q-Morph-inspired bounded front: local side selection, edge recovery/insertion, cavity replacement, collision/no-progress handling, transition templates and residual T3 closure.
- TinyAD local patch optimization with safeguarded line search and independent final validity checks.
- Independent global validation, qualified S3 admission and final component publication.
Proposed modules (verify before creating):
src/anymesher/quad.py
src/anymesher/_quad/{__init__,driver,contracts,orientation,layout,counts,transitions,validate,diagnostics,reference}.py
src/anymesher/_quad_count_worker.py
src/anymesher/quad_native/{CMakeLists.txt,python_module.cpp,mesh_state.*,front.*,transitions.*,quality_energy.*,count_adapter.*,third_party_attribution.*}
docs/{QUAD_FIRST_DESIGN,QUAD_FIRST_REUSE,QUAD_FIRST_QUALIFICATION}.md
tests/quad_first/
benchmarks/{quad_first_cases.py,quad_first_measure.py,compare_quad_methods.py}
reports/quad_first/<task-id>/WORK_STATUS.md

Public API:
- Add separate versioned QuadMeshingOptions and quad_options=None to appropriate existing hybrid/surface entry points.
- quad_options=None must execute the untouched old path.
- Initial supported route: planar linear shell faces, holes/concavity/shared structural interfaces, quad-dominant closure with qualified triangles.
- Reject contradictory recombine=False, explicit Frontal-Delaunay+quad-front combination, unsupported anisotropy/curved/higher-order cases in the first route unless a caller explicitly permitted one bounded old-route fallback.
- Native quad library absence is a capability error for explicit requests; broken-present libraries/invariant failures/cancellation are not treated as absence.

Construction:
1. Freeze component/source revision, owners, charts, loops, protected curves, exact shared boundary IDs/stations.
2. Build one usable size-informed CDT seed; initial O(N) triangulation is allowed once, not per front step.
3. Build planar cross field, initially z=(cos 4theta,sin 4theta), boundary tangent/explicit direction anchors, deterministic sparse smoothing, low-confidence diagnostics.
4. Maintain resident mixed local topology; no full export/sort/copy per front edit.
5. For each front edge, deterministically rank bounded proposals; prefer existing side/top edges, then local recovery, then bounded insertion; stage local cavity/Q4/T3/front change; validate; optionally optimize; commit atomically or rollback.
6. Explicit all-candidates-rejected/no-progress path: transition, qualified closure, typed unsupported, or one explicitly allowed fallback.
7. Prove at least one fixture where the accepted quad construction requires genuine edge recovery or node insertion and cannot be reproduced by simple pairing of the initial triangles.
Q0 — Freeze contracts/reuse:
- Recheck current main delta and worktree ownership.
- Audit donor/library code/licences; resolve Eigen immutable pin/hash.
- Freeze QuadMeshingOptions schema, fixtures, quality definitions, resource guards.
- Build scoped native prototype without altering old _native ABI.
- Run actual tiny libSatsuma and TinyAD smoke calls if feasible.
- Commit design/reuse/manifest evidence.

Q1 — Mixed state and port primitives:
- Implement local residual-T3/accepted-Q4 state, generations/stale-handle protection, transaction journal, local incidence/front records.
- Port only eligible Q-Morph front classification/candidate/recovery primitives.
- Independent local cavity validator, cancellation/rollback, all-rejected handling.
- Tiny tests proving no fake quads and no whole-mesh export for a local edit.

Q2 — Working planar front / M1:
- Boundary-driven advancing rows on a nontrivial off-centre-hole plate and concave fixture using one valid seed.
- Add progress budgets, collisions and residual T3 closure.
- Demonstrate real recovery/insertion, exact protected boundaries and bounded runtime.
- Fresh-context review and fix material findings before internal M1 commit.
Q3 — Guidance and transitions:
- Add separate planar cross-field guidance and conflict/low-confidence diagnostics.
- Add qualified spacing-change, collision and closure transition templates with rotation/reflection tests, explicit preconditions and parent-transition replacement records.
- Compare guided vs unguided on frozen small fixtures; do not introduce a general parameterization solver.

Q4 — Real Bi-MDF integration:
- Model only representable private patch/corridor count problems; fixed structural boundary counts/stations remain hard constraints.
- Use actual libSatsuma solver through an isolated/private adapter or worker as needed.
- Validate returned integer solution against original constraints independently.
- Add infeasible/invalid/equal-cost deterministic/cancellation/timeout/worker-lifecycle tests.
- Show at least two nontrivial end-to-end count applications. Smoke-only does not count as Q4 delivery.

Q5 — TinyAD local optimization:
- Implement documented Q4 size/alignment/shape/Jacobian energy for free local interior nodes only.
- Numerical/analytic derivative checks, finite-domain guards, safeguarded line search, rollback on worse/invalid step and neighbor-halo validation.
- Demonstrate accepted quality improvement and measure overhead; protected nodes/stations never move.

Q6 — Public structural integration / M2:
- Complete API/schema/serialization/capability reporting, owner associations, component transactions, S3 admission, automation hooks and minimal GUI selector only if already fitting architecture.
- Mixed mapped/front components, reversed shared edge, three-sheet declared junction, material/thickness partition, beam attachment, cancellation-before-publication and unsupported request tests.
- Existing defaults/old persisted models unchanged.
- Production installed native path must actually exercise Q-Morph-derived front logic plus libSatsuma/TinyAD when requested.
Q7 — Qualification / M3:
- Tiny operation/contract tests: front/recovery/insertion, cavity commit/rollback, all-rejected/stale/no-progress, protected geometry, holes/multi-holes/concavity/narrow ligament/invalid overlap/degenerate input, parity, every transition template, orientation 90-degree equivalence/conflicting anchors/low confidence, Bi-MDF brute-force checks, TinyAD derivative/rejection checks, native boundary, worker lifecycle and scope guard.
- Engineering assemblies: off-centre hole, concave/narrow ligament, mapped+front adjacent reversed shared edge, declared three-plate intersection with identical interface IDs, material/thickness + beam contracts, cancellation after construction but before publication, typed unsupported curved/higher-order request.
- Mixed Q4/S3 solver-facing downstream checks at meaningful physical resolution: isolated triangles, chains, boundary fill-ins, clusters, roughly 1-25% triangle mixes where existing accepted consumer corpus permits; no new S3 formulation here.
- Source + installed-wheel qualification on supported platforms; compiled/source parity on small cases; exact source/native/dependency/artifact identities.
- No tag/release/PyPI/default promotion.

Quality/performance:
- Freeze nominal corpus before tuning.
- Proposed nominal target: >=85% active Q4 by count and area on designated regular planar front fixtures; >=75% on designated transition-heavy fixtures. Hard quality/ownership/station/coverage/S3 failures always fail regardless of Q4%.
- Report active counts, Q4 fraction by count/area, orientation/alignment metric, element quality, DOF where useful, accepted front/recovery/insertion/template work, runtime, RSS and fallback.
- Old-route regression checks ensure adding quad package adds no work to old calls.
- Method-vs-method may have different connectivity; compare same geometry/sizing/boundary/quality intent, not equal hashes.
- Formal timing only at coherent checkpoints: one warmup + seven unprofiled isolated measurements. Profile separately.
- Routine development: tiny to about 1k-10k active elements. Selected <=100k checkpoint only after smaller gates and only if useful. Never run 500k/workstation/equivalent renamed/aggregate workloads. Never use the slow 10k-request hole diagnostic as an ordinary gate.
- No hidden repeated full-face retries. Local edit complexity counters matter more than giant benchmark runs.
OpenCode operating rules:
- Work directly in this worktree/session. Do not spawn subagents for broad repository exploration; inspect directly to avoid the prior stalled explore-agent path. Fresh-context review may be a separate bounded reviewer only after a coherent implementation milestone.
- Update reports/quad_first/full-programme/WORK_STATUS.md after every meaningful milestone, test batch or decision.
- Commit coherent accepted changes on opencode/quad-first-v1. Do not push, merge main, create PR, tag, release, publish, reset, force-push or destructively clean.
- Preserve unrelated files. Do not modify the primary C:\Github\ANYmesh worktree.
- Investigate routine failures autonomously. Do not weaken/delete/xfail tests merely to go green.
- Keep build/dependency fetches pinned and reproducible; do not silently track upstream branches.
- Formal benchmark runs must be isolated from competing local inference/process load.
- At each M1/M2/M3, do a fresh-context review of diff/contracts; fix material findings and rerun affected tests before continuing.
- Continue through Q0-Q7 without waiting for supervisor approval between internal gates.

Final handoff must report:
- starting/final HEAD and commits, changed/uncommitted files, preserved pre-existing work;
- actual model/reviewer identities;
- exact donor routines adapted and actual libSatsuma/TinyAD production-path use;
- architecture decisions/deviations;
- exact build/test/benchmark commands and terminal pass/fail/skip/interrupted counts;
- source/wheel/platform evidence and artifact/dependency pins;
- active counts/Q4 fractions/front work/timing/RSS for measured fixtures;
- fresh-review findings and fixes;
- unsupported/deferred cases;
- confirmation 500k excluded and no publication/default promotion performed.

Start now and continue until the full programme is complete or a genuinely critical blocker as defined above is reached.