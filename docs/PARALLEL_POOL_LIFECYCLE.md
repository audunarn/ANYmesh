# Parallel pool lifecycle and verification

This opt-in ANYmesh prerequisite supports ANYfem's component-parallel integration. It preserves native meshing algorithms and application defaults. It does not qualify a solver formulation, replay SG1/RA1, or enable application parallel execution.

## Public API

```python
from anymesher import (
    ParallelOptions, ParallelPoolLease, create_parallel_pool,
    generate_hybrid_mesh_parallel, generate_hybrid_mesh_result_parallel,
)

with create_parallel_pool(2) as lease:
    mesh = generate_hybrid_mesh_parallel(
        model, target_size=0.25, order="quadratic",
        parallel=ParallelOptions(pool_lease=lease),
    )
```

`create_parallel_pool(workers=None, *, name=None, start_timeout=...)` returns a caller-owned exclusive lease. Both generation functions require `target_size` and accept `parallel=ParallelOptions(...)`; the result wrapper returns `HybridMeshResult`, the other returns its mesh. Options retain `workers`, `min_components`, `pad_factor`, `executor`, and add `pool_lease` and `min_estimated_elements`.

A successful borrowed pool stays reusable. Cancellation or failure invalidates it; the owner creates a new pool for a later request. Lease state changes are locked. Borrowing concurrently or closing an active lease fails clearly. Without a lease, the library owns a cold pool and disposes it before returning a result.

Bare/shared/unknown executors fall back before submission. Passing both executor and lease is rejected. There is no thread executor: the existing whole-model chart hint is process-global. Face/member/beam selections are normalized once before refusal or dispatch, preserving explicit empty selections and one-shot iterators.

## Failure and containment

Component work is polled at100ms intervals. A shared cooperative signal reaches the existing engines; cleanup uses one5s termination/join budget. Original owner/cancellation/deadline errors propagate, with cleanup evidence attached. A crashed worker fails without serial retry. Startup containment refusal may fall back only before work and only after successful disposal. Failed cleanup never publishes a mesh. Unknown executors and unrelated processes are never terminated.

Windows workers enter a kernel kill-on-close Job Object before task admission. The SDK extended-limit information class is9, and its64-bit structure is144bytes. Job creation, arming or assignment failure fails closed. Windows units exercise arming, assignment refusal, cancellation, descendants and abrupt owner death.

POSIX workers use verified private process groups. An independent guardian watches parent and worker death sentinels; it survives abrupt worker death long enough to terminate that worker's owned group. Startup admission acknowledges the guardian before tasks. Group cleanup includes descendants of already-dead workers. WSL units exercise both abrupt parent death during a task and abrupt worker death with a sleeping descendant.

Transport payloads are serialized before queue submission. Untransportable arguments, results or exceptions fail promptly instead of disappearing in a queue feeder. Fresh owner partition validation checks content at dispatch and join; stale, foreign and altered bindings retain typed owner errors. There is no partial-result publication.

## Worker resource priority

Every worker applies a best-effort resource-priority step once at its own startup; the parent/GUI process priority is never touched. Windows keeps the existing below-normal priority class semantics and legacy `priority_applied`/`priority_class` fields. POSIX workers now apply a measured nice floor of5: the current nice value is read (`os.getpriority`), the request is `max(current,5)`, `os.setpriority` runs only when current nice is below5, and the actual value is read back. An already-lowered process is never raised back up, and warm reuse never accumulates increments. `applied` is true only when the effective value was measured at or above the floor; a no-op on an already-lowered process reports status `already_lowered` instead of claiming a syscall. Missing APIs, permission refusals, readback failures and below-floor readbacks are explicit statuses with error text; no nice value is ever invented.

This evidence is exposed only under `parallel.runtime.worker_controls` (`nice_before`/`nice_requested`/`nice_effective`, `priority_mechanism`, `priority_status`, `priority_error`) alongside the legacy fields. It is a new candidate: the earlier WSL calibration at17754aa binds the frozen old profile only (no Linux priority application existed there), so it does not measure this change, and no new Linux performance claim is made. Defaults remain off and main holds.

## Diagnostics and routing

Deterministic route/reason/component semantics remain in `mesh.hybrid_diagnostics["parallel"]`. Machine-dependent data lives in its `runtime` child: pool kind, worker count/PIDs, timings, lifecycle, working-set samples, measured priority and numerical-library thread counts, and component estimates/basis. Consumers must exclude `runtime` from semantic cache identity. Unavailable measurements are explicit; working-set samples are not peak-memory claims.

Mapped side-length products and member/edge lengths yield labelled heuristic estimates; supplied seeding uses its existing divisions. Unsupported owner measurements yield unknown, never invented area or zero. An explicitly selected `min_estimated_elements` threshold routes small/unknown estimates serially. No automatic threshold or application default changes here.

The join follows fixed source-component order. Exact mesh IDs/connectivity are independent of worker count; serial and component numbering may differ. Source ownership, quadratic order and coupling weights/eccentricities remain preserved.

## Development verification2026-10-07

Inputs: ANYmesh base44f9bd11c704da05936019f6deb36c167b46045a; archived ANYgeometry660430a5041a8315c523bdb542d64271f9445f7a. Owner wheel provenance SHA2565adbcfdd3a6df9d897160cd1f8217379c8bf157e711182a68ba260f03fa23956. WindowsPython3.14.2 and unchanged cp314 native binary SHA256ab6b5597b4f421f302e435ba143dc58c92f20faba82b02d5e98c84d5fbbee754; numerical-library threads1.

- Windows pool and independent refusal-selection units:24passed/1POSIX-onlyskip, precise12.7164044s process wall (`attempt-20261007T102538308`).
- Windows component/lifecycle suites:45passed and one fault-injection fixture teardown failed,9.2069639s (`attempt-20261007T102552849`). The fixture restored the deliberately failing cleanup method before final disposal; its focused rerun passed. Other passed cases were not repeated merely to obtain a single green summary.
- Independent exact quadratic IDs/coordinates/connectivity/coupling identity across1/2workers and unchanged source: passed,4.1912758s (`attempt-20261007T102942775`). An initial Root probe used an unsupported integer order; that failed evidence is preserved.
- Independent source-change-at-join refusal and corrected cleanup-error fixture:2passed in the3.6939768s run containing that initial invalid-order probe (`attempt-20261007T102837724`).
- Independent WSLUbuntu24.04/Python3.12 pool-only suite:18passed/3Windows-onlyskips,38.9334193s (`attempt-20261007T101630811`). Exact hosted owner wheel installed. This evidence covers the unchanged POSIX guardian branch; later Windows-only ABI/test changes do not replace it. It does not establish Linux native mesh or ANYfem parity.

Raw commands, exact process-wall accounting, source hashes, stdout/stderr and all earlier failures are retained locally under `reports/anyfem-parallel-lifecycle/`. Root corrected malformed flags in its initial worker brief using saved command records; the original ledger is retained. No scientific budgets were renewed. ANYmesh real-meshing allocation is600s total,360s worker share,120s per attempt; the ledger records consumption and remaining authority.

Final owner-error/cancellation identity regressions:3passed,2.9006955s (`attempt-20261007T103531339`); changed public stale-source join refusal:1passed,2.1424241s (`attempt-20261007T103538986`). Real-meshing consumption57.9388029s total (worker47.9111262s); remaining542.0611971s, with Root's independent allocation respected. The final3.14/3.12 module-location and thread-count probes are also saved. Earlier SDK, test-fixture, iterator and Root-probe input failures remain available; their causes were corrected without relaxing checks.

Root independently reviewed the Mistral implementation and reconciled the SDK, containment, exception, transport and public-boundary findings. Final binding regression results and source pin are recorded in the living task note and coordinator handoff. Runtime observations establish no speedup claim. ANYfem GUI/installed-artifact/native Linux parity and performance acceptance remain the integration owner's work; defaults remain off.

Cross-platform worker priority candidate2026-10-07: focused unit run `tests/test_parallel_priority.py` on WindowsPython3.14,14passed/1POSIX-onlyskip,32.4394304s process wall (`attempt-20261007T120311494`). Covers pure-mock POSIX nice-floor cases (0->5 readback, already-lowered no-op, no accumulation, missing API/permission/readback truthfulness), the Windows dispatcher legacy-field passthrough, and a real1-worker spawn/warm-reuse pool with unchanged parent priority class. Root subsequently removed fixed15s/job waits from the test probe; affected Windows warm-reuse test passed in2.4947758s (`attempt-20261007T120645327`). These are unit-check observations, not meshing speed measurements.

Independent WSLPython3.12 focused suite:13passed/2Windows-onlyskips,8.6621802s (`attempt-20261007T120734755`). Stronger OS-readback correspondence checks then passed for normal inheritance (parent0unchanged, worker0->5; two jobs on the same PID at5),8.1734701s (`attempt-20261007T120929651`), and genuinely inherited lower priority (parent8unchanged, worker8preserved; two jobs on the same PID at8),5.9773408s (`attempt-20261007T120942720`). The measured startup payload matches the actual worker queries. No native meshing, calibration, caller numerical-budget consumption or renewal occurred for this delta. Old17754aa performance evidence remains bound to its original source; main stays on hold and defaults stay off.
