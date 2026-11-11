# ANYmesher source integration gate — 2026-10-28

**Unmet.** No full-suite pass or inferred timeout counts. The candidate production
provider stays disabled; baseline failures do not waive the release gate.

The root-owned slice had900 supervised process seconds maximum, each process
at most600s. All terminal receipts below verify owned cleanup; no historical
scientific campaign or failed native generation was replayed.

| Check | Observed result | Process-wall seconds | Evidence directory |
|---|---|---:|---|
| Build both repository quad workers | pass |13.304323844|mesher-quad-workers-20261028-b7a8d20d70|
| Initial full-suite harness | no pytest in FE runtime |0.072932978|mesher-full-source-20261028-c25706b02e|
| Full source suite in existing test runtime |2 collection errors,3 skips; ANYfem absent |18.744680483|mesher-full-source-runtime-repair-20261028-9a74c528c6|
| Complementary suite excluding exactly those two consumer modules |timeout; no JUnit/terminal counts |600.044542679|mesher-source-without-missing-consumer-20261028-e948d92f5d|
| Complementary fail-fast diagnostic |144 passed,1 failed,3 skipped |15.790989558|mesher-source-failfast-diagnostic-20261028-9f0c610469|
| Exact remote-baseline comparison of first failure |1 failed, same assertion |7.626715650|mesher-baseline-q4-recovery-20261028-00f3b3dec0|

Evidence root: `C:/ANYworkspaceWork/workflow2-20261027/evidence`. Each check's
`identity.json` and `process/identity.json` retain the exact command, candidate
source hashes, runtime/one-thread environment and timeout. Full logs remain
in `process/stdout.log` and `stderr.log`; terminal supervisor results retain
precise process-wall and cleanup. Commands used the existing
`tools/workflow2_check.py` supervisor with WSL Ubuntu-24.04 and private Linux
Python3.12 runtimes. Both compiled quad workers were required. No optional
ANYmesher C++ extension was present; no compiled-core claim is made.

The missing collection prerequisite affects
`tests/test_authored_child_project_cells.py` and
`tests/test_authored_project_references.py` (11 test functions). They require
a real pinned ANYfem consumer and its headless dependency closure. ANYmesh's
existing dev metadata and CI do not install that consumer. Read-only source
inspection found the relevant owner symbols compatible, but runtime import
compatibility is unproved. Do not stub the consumer or silently replace the
candidate geometry with a different dependency. No sibling edits were made.

The first actionable complementary failure is
`tests/quad_first/test_q2_recovery.py::test_q4_t3_recovery_is_conforming_and_area_preserving`:
`no Q4/T3 interface edge admitted a conforming recovery`.
An isolated disposable export of both recorded baselines reproduced the exact
same failure: ANYmesh0415256ee5e0a0a6088465f58bec7df3de3f2e3e and
ANYgeometry72a7e011fe3777a6aa8107cae388e3a36a917b39. Source archives and their
selected baseline source hashes are retained in `mesher-baseline-comparison-20261028`
and the comparison receipt. No quad/recovery/worker source was changed by W2.
This establishes a pre-existing failure in this environment, not a waiver or
classification of the other incomplete-suite failures.

Supervised check-process total: **655.584185192/900s**. Disposable archive
preparation was ordinary filesystem/Git read work, not a numerical check; its
tool-wall receipt is separate. Remaining process authority was not spent on
another broad run after the timeout/baseline comparison. Full source/native/
installed release/macOS gates remain unresolved. Review: root inspected raw
failure traces and baseline result; independent read investigation assessed
consumer prerequisites, without running tests or granting acceptance.
