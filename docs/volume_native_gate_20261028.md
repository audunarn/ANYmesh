# Installed native gate — 2026-10-28

State: **implemented and focused checks verified; full gate NO_GO / incomplete**. Root reconciled the finite outcomes and preserves the full gate failure. This finite evidence concerns Ubuntu24.04 Python3.12 only. It does not promote the public volume provider or qualify TetWild, FE physics, other platforms or a release.

## Candidate and method

The runner `tools/workflow2_native_gate.py` copied immutable package sources into `evidence/native-core-gate-20261028-a1/sources`, built required native wheels there, and installed them without dependency resolution into a private clone of the proven real-consumer runtime. Live production source was never built in place or edited. The real ANYfem consumer remains immutable commit `ed845899e7c358b46376bc3ad873327b0267e174`;165 installed Python files match the prior export proof, with zero differences.

`manifest.json` freezes343 package files,269 test files, compiler identity, dependency closure, both required quad workers, selection, runner and supervisor. All269 test hashes remained unchanged during execution. The live THIRD_PARTY_NOTICES.md changed concurrently for root's exact-provider LGPL notice; frozen wheels retain their original notice/hash. No package code changed.

The private installed modules were proved under `-I`: geometry `_core.abi3.so` and mesher `_native.cpython-312-x86_64-linux-gnu.so`, with geometry reporting native. All package and consumer origins were private site-packages. Native require flags were1 and disable flags0; thread variables were1; the existing MCF/TinyAD worker paths were absolute and required.

| Artifact | SHA256 |
|---|---|
| anygeometry-0.4.6-cp311-abi3-linux_x86_64.whl |348ca633a53220e480464c754a3e2722c9ebcb98ea885629e128a67516dfeea8|
| anymesher-0.5.1-cp312-cp312-linux_x86_64.whl |73598e75a23f1597bf35d97bd19f9d8dbe9dd9085e8e8bdd580651efbdadd1a2|

These retain existing development version policy; they are not release artifacts. `artifacts.json` and `installed-origins.json` include wheel sizes, module hashes, compiler/runtime and installed versions.

## Execution and evidence

Every attempt used root's `tools/workflow2_process_supervisor.py::run_linux`, unique output directories, strict machine JSON and a shared1800-second process-wall ledger, with180 seconds reserved. Exact commands, effective environment and candidate/runner hashes are in each attempt's `identity.json`; raw stdout/stderr and terminal cleanup receipts are in `process/`. Every cleanup was verified. `budget.json` records1247.857280693 seconds used and552.142719307 remaining. No authority from earlier closed campaigns was reused.

| Check | Observed outcome | Process-wall seconds |
|---|---|---:|
| Source/runtime freeze | completed |218.094341678|
| Initial geometry build | failed: missing setuptools.build_meta |9.668780558|
| Pinned disposable build prerequisites | completed |3.543160563|
| Geometry native wheel build | completed |20.622782565|
| Mesher native wheel build | completed |30.843873189|
| Install native wheels, no dependencies | completed |1.966722044|
| Isolated installed-origin/native probe | passed |6.272898079|
| Initial focused test run | pytest DrvFS fd-capture failure |4.846359767|
| Focused rollback/journal tests with Linux-local temp |4 passed |4.164588993|
| Existing geometry native/parity contracts |153 passed |14.947868502|
| Full installed collection |4129 selected;0 errors;0 deselected; optional gmsh module skip |14.317743094|
| Full installed suite | timeout; partial1155 passed,29 failed,3 skipped |900.053781692|
| Python-geometry causal diagnostic | completed |3.920130389|
| Native-geometry causal diagnostic | completed |2.489838876|
| Child-only recovery-search ablation | completed |2.358868519|
| Explicit existing-layout diagnostic | completed |2.307359480|
| Previously unreached nativeT3 contracts |85 passed;0 skipped;0 deselected |7.438182705|

The two conforming final-front-stage rollback cases exercise actual staged retile/front advancement followed by forced error or actual cancellation. The production algorithm is unchanged. The two journal tests prove flushed failure detail and retained active phase without a false completion claim.

The initial build failure and fd-capture failure remain preserved. The build repair added pinned setuptools80.9.0/wheel0.45.1 only to the disposable runtime. The capture repair changed temporary descriptors to owned Linux `/tmp/anymesh-native-core-gate-20261028-a1`; outputs remain in evidence. `harness-amendment-1.json` retains old/new runner hashes. Geometry test selection was prospective, but individual selected geometry test hashes were recorded during its first run; all package and mesher test hashes were frozen before test runs. This timing limitation is not concealed.

The full command, from the ANYmesh repository, was:

```text
<private-runtime>/bin/python -I <ANYmesh>/tools/workflow2_native_gate.py pytest-child -- -q -o pythonpath= --confcutdir=tests --import-mode=importlib tests --durations=25 --junitxml=<evidence>/full-installed/pytest.xml
```

The supervisor bound was900 seconds. No tests were omitted or deselected. The repository conftest appends fixture/repository directories, not src. The live journal flushes every setup/call/teardown phase, outcome, duration and complete failure longrepr. After timeout the controller reconstructs active-node and partial-count evidence without declaring a full pass.

## Unmet gate and limits

The timeout left1187 completed nodes, one active call and2941 not started from4129 selected. Partial completed counts are1155 passed calls,29 failed calls and3 skipped nodes. There is no terminal JUnit XML or session-finish event. Active at kill was `tests/test_cylindrical_quadratic_hybrid_trial.py::test_work_bearing_quadratic_trial_from_real_hybrid_seed[closed-T6]`. `full-installed/journal.jsonl`, `live-summary.json` and process receipts retain the complete available evidence.

Failure groups include18 curved-quad tests,10 planar-quad/growth tests and one authored preparation-provenance test. They are not collectively classified as benign count drift or as W2 defects. No golden tuples, shape/area/coverage tolerances or assertions changed.

Three skipped nodes comprise two historical authored handoff tests for a genuine ZIP-wheel loader with three exact pinned owner-wheel hashes, plus a compiled-absent fallback test that skips because the extension is present. The historical ZIP-wheel handoff contract is not applicable to the current extracted candidate wheel; its skip reason mentioning source checkout is misleading, but its loader/pins remain unchanged. That historical gate was not exercised or waived. Optional gmsh collection skip is separate. Compiled triangulation/recombination contracts executed before timeout. Later nativeT3 contracts were not reached in the full run, then executed under the separately approved focused selection.

## Causal diagnostic and next decision

Root approved three bounded diagnostics after the full run terminated. Frozen helper/plan/amendment artifacts are retained. All three produce the same representative ruled h.6 mesh: linear18 nodes/11 Q4/2 T3; quadratic48 nodes/11 Q8/2 T6. Native and Python geometry seed/final chart states are exact JSON-equal. The seed already has18 nodes/24 T3 and no successful recovery occurs. The observed mismatch therefore is not explained by compiled geometry or the new conforming recovery search for this representative case; other failures remain unadjudicated.

Unchanged production validation gives chart area/reference2.0766559657295187 and ratio1; min Q4 scaled Jacobian0.6575642519946511; min T3 angle41.290682 degrees. Independent Shapely chart footprint symmetric-difference and overlap areas are0. Support residual is<=2.23e-16 and all Q8/T6 mapping certificates are positive. Validity does not establish expected all-quad closure.

Source inspection finds hybrid.py defaults explicit quad_options to adaptive layout. The existing hardcoded topology references may describe explicit existing layout. Baseline exported hybrid/seed/options/driver source hashes equal the current candidate. Root prospectively approved one explicit existing-layout diagnostic (<=30s) and previously unreached nativeT3 contracts (<=120s), within the same1800-second ledger; `prospective-amendment-2.json` freezes inputs/helper/selection before execution. Existing layout restores the representative CH11 references exactly: linear15nodes/8Q4/0T3; quadratic37nodes/8Q8/0T6, with zero recoveries. Unchanged quality/mapping/coverage predicates pass; chart symmetric-difference is0 and floating overlap residual is-4.44e-16. This distinguishes layout-specific reference evidence for the one case, not a blanket resolution of other failures or permission to relax adaptive all-quad requirements. Five installed-native T3 runtime/storage/persistent-membership/capsule-ABI/cancellation modules passed85 tests, zero skips/deselections, with terminal JUnit and phase journal. Total completed focused checks are242 passes across distinct focused selections; those are separate from the partial full-run counts. No second full run, production repair or reference update was launched. Root retains profile decisions, independent review, reconciliation and acceptance.

## Root-adopted reference-policy follow-up

This uses the SAME1800-second ledger. Before editing, eleven original reference files and their old hashes were copied to `reference-policy-amendment/original` and `before-edit.json`; every old hash matches the original full-gate manifest. New hashes, case-policy comments and exact selection were frozen in `after-edit-before-run.json`. Only legacy seed/topology/MCF/Q5 fixture configuration changed to explicit `layout_policy="existing"`. Original assertions, count references, tolerances, refusal-only calls and IS1 adaptive files remain unchanged. CH9 reuses the same-argument existing-policy CH4 generator without editing CH8. Production default and installed wheels are unchanged. A new tiny PQ3 test proves omitted policy with explicit quad options matches explicit Adaptive topology/nodes/stations and reports Adaptive.

The once-only eleven-file focused command is fully recorded in `reference-policy-focused/identity.json`. It completed in23.050217171 process-wall seconds:73passed,9failed,0skipped; terminal JUnit and flushed traces are retained. The new default-Adaptive check passed. Nine remaining failures are CH9 coarse/fine/certificate references, CH11 graded ruled/coons references, CH12 conical owner reference and three PQ6 MCF/front-work references. Examples: cone41nodes/27Q4/6T3 versus37/23/6, fine cone127nodes exceeds unchanged124 cap, graded36/26/4 versus34/24/4, MCF baseline53attempts versus47 and public45Q4 versus37. No replacements were blessed.

Two direct enabled/disabled PQ6 calls outside `_public` initially retained the Adaptive default. They received the same approved explicit Existing input correction, with prior/new hashes saved before the unique focused recheck. That recheck completed5.004728153s and failed1 test: MCF APPLIED/status checks now pass, but unchanged downstream topology references still fail. This is not resolved by further fixture policy changes; no production repair or golden update was attempted. Full traces are retained in `reference-policy-direct-caller`.

Follow-up supervised checks used28.054945324 of the allocated240 seconds. Whole-ledger use is1275.912226017 seconds, remaining524.087773983 including180 reserved. All process cleanup was verified. The original full NO_GO journal/JUnit absence/partial results remain unchanged; no full rerun occurred. `reference-policy-amendment/handoff.json` freezes all eleven final test hashes, original assertion-line preservation, remaining failures and authority. Adaptive boundary, deepcopy/protected-station and slow closed-T6 issues remain outside this edit and unmet. State: fixture configuration implemented and partly verified; root and independent non-author same-family Read review confirmed preservation of assertions/tolerances/goldens and the unresolved failures.

Root parsed the full phase journal separately: all335 selected `tests/quad_first/`
nodes completed with passed calls, zero failed calls. That prefix supports the
Q4 repair but cannot qualify the uncompleted suite. The installed dual-native
profile is stricter than current ANYmesh CI's pure-geometry/native-mesher profile;
no profile switch or waiver was made. After campaign closure, only the runner's
console summary was compacted; its original controller is preserved as
`controller-before-compact-output.py`. No production or claim-bearing run was
repeated for that output change.
