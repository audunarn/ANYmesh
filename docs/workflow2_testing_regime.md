# Workflow 2 focused testing and evidence

Use affected contracts during development, then the applicable full installed
gate at integration. Native mesher qualification is a separate registered study;
ordinary pytest checks neither consume nor grant native meshing calls. Do not
repeat passing evidence unless its relevant inputs changed or a concern remains.

`tools/workflow2_check.py` records the exact command, source identities, effective
thread/native flags, full stdout/stderr, JUnit where applicable, process-wall time
and cleanup receipt. Supply explicit source roots and the intended Python
interpreter; the runner does not install dependencies or select a test profile.
Keep unique output directories under the shared evidence root.

```powershell
& ..\runtime\Scripts\python.exe tools\workflow2_check.py `
  --name volume-contract-dev --cwd (Get-Location) `
  --evidence-root ..\evidence --source src\anymesher\volume.py `
  --pythonpath src --pythonpath ..\ANYgeometry\src `
  --timeout 180 -- python -m pytest -q tests\test_volume_mesh.py
```

This example is a command template, not a claim that it was rerun. Use the
existing affected-test selector and smoke profiles for other changed subsystems.
Q4 changes warrant `tests/quad_first`; packaging changes warrant
`tests/test_exact_provider_notices.py`. Thin metrics have controlled tests in
`tests/test_thin_volume_qualification.py`; they do not prove a generated mesh.

On Linux, the bundled `workflow2_process_supervisor.py` owns the descendant tree,
including children that create new sessions, through subreaper and pidfd APIs.
It refuses launch when this controller capability is absent. Use a capable
controller (the registered Linux 3.12 runtime) to supervise other supported child
interpreters. This is not a product Python ABI requirement. The legacy optional
`--linux-supervisor` override must expose this descendant-aware `run_linux`
contract; a session-only helper is refused. Windows retains the owned Job Object
supervisor. Uncatchable controller termination/host failure is outside the
userspace cleanup guarantee.

Current controlled evidence: `workflow2-cleanup-controller-contract-0c86bf8eb8`
contains **6 passed, 0 failed/error/skipped**, 6.488349426 process seconds.
The exact pytest selection is `tests/test_workflow2_process_supervisor.py`;
cases exercise timeout, cancellation, actual SIGTERM, the real generic CLI and
source-gate callers, and missing-capability refusal before launch. Receipts
confirm zero remaining owned descendants. Earlier two-caller evidence is retained
at `workflow2-check-caller-regression-cd189f728a`. Independent non-author same-family
Read review checked the actual caller integration and final source identities.

The source repair campaign, native installed campaign, thin study and notice-only
campaign each have separate immutable manifests, budgets and failure histories.
Their exact commands/counts are linked from the project record. In particular,
the native full suite timed out with failures, and the rotated thin fixture
failed its frozen sampling bound. Focused passes do not waive either gate;
unused elapsed-time budgets do not renew native call authority.

Terminal root reconciliation (static only, no tests/provider imports/native
calls): `remaining-gates-root-reconciliation-field-map-fixed-312f2b5ab8`,
0.906000000017 seconds. It verifies current frozen production/runner/supervisor
hashes, parses the native journal and controlled-test XML, and validates changed
Python syntax/JSON. The earlier 0.296999999991-second harness failure compared
the shared reference helper's hash with the thin runner by mistake; original
script/receipt are retained at `remaining-gates-root-reconciliation-97a8fb44a2`.
Correcting that field mapping required no scientific execution or evidence edit.
