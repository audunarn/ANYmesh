# Plan: using the parallel component route from ANYfem

Implementation checkpoint2026-10-07: ANYmesh's opt-in public pool/lease
prerequisites are implemented and reviewed. Current API, focused Windows/WSL
evidence and unresolved application acceptance boundaries are in
[PARALLEL_POOL_LIFECYCLE.md](PARALLEL_POOL_LIFECYCLE.md). The original proposal
below remains historical context; ANYfem integration proceeds in its own
checkout, with parallel execution off by default.

Date 2026-10-07. Status: **plan only; nothing in ANYfem was changed** (it has
other workers' uncommitted edits). Written from ANYmesher; the ANYfem owner
decides. Evidence and background: `docs/PARALLEL_MESHING_STUDY.md`,
`benchmarks/parallel/exp8_anyfem_project.py`.

## 1. What is already established

- ANYmesher local main (5814510, not pushed) has
  `generate_hybrid_mesh_result_parallel` (opt-in). It meshes components that
  ANYgeometry's `plan_independent_components` certifies as independent in a
  spawn process pool and joins them. Anything not certified falls back to the
  serial route with the reason in `mesh.hybrid_diagnostics["parallel"]`.
- It now accepts the argument pattern of ANYfem's whole-project call
  (`Project.generate_mesh`, `model/project.py` around the
  `generate_hybrid_mesh(working, **meshing_parameters)` call): `working_copy`,
  selections naming every face and member, `interactive` certification without a
  change set, a precomputed `seeding`. It refuses `strict` certification,
  `change_set`, `audit_policy`, `quad_options` (quad-first) and partial selections.
- Through that call, with ANYfem unmodified and the function monkeypatched in,
  `strategy="native"` meshes are identical to serial up to numbering. Plates of
  about 700 elements: 16 plates 17.7 s to 5.6 s (3.1x), 32 plates 35.6 s to 9.7 s
  (3.7x), cold pool each call. Plates of about 70 elements are slower in
  parallel (spawn cost). Measured on a shared 32-core machine, single runs.
- ANYfem's *mapped fast path* (already structured faces, no beams, no
  preparation changes) never calls the hybrid mesher. It is not affected and
  needs nothing.

## 2. Facts that shape the design (all checked in source or by a run)

1. **Mesh jobs run in a daemonic worker in the GUI.** `JobManager(executor="process")`
   is the Qt default (`ui/qt/app.py`); `job_process.py` starts the worker with
   `daemon=True`. A process pool cannot be created from a daemonic process
   (`AssertionError: daemonic processes are not allowed to have children`, run
   to confirm). So with the GUI default the route cannot work as is.
2. **Orphans.** `JobManager.shutdown()` and cancellation `terminate()` the worker
   only. Pool children of a worker would survive it. Any design needs tree kill
   (Windows Job Object) or a parent-owned pool.
3. **ANYfem prepares geometry itself** before the hybrid call and remaps the
   result afterwards (`preparation`, `closure`, `remap_mesh_to_source`). The
   route receives and returns ids of ANYfem's `working` geometry, so it composes
   with that without change.
4. **Hashing.** `mesh_semantic_hash` hashes the codec form of the mesh (numbering
   included) plus `hybrid_diagnostics` minus keys ending in `_seconds` and a few
   id keys. `mesh_hash` binds stored meshes to solve jobs, so route-dependent
   numbering is fine, but any machine-dependent value in `hybrid_diagnostics`
   (today `hybrid_diagnostics["parallel"]["workers"]`) would make the hash depend
   on the machine.
5. **Cancellation.** ANYfem cancels cooperatively through
   `cancellation_check`. The route calls it only between component completions,
   so latency is the longest component, and `shutdown(wait=True)` also waits for
   running components.
6. **Single-face chart quirk** (fixed inside the route, see study note):
   a plate meshed alone differs from the same plate in a multi-face model.
   Whether that face-count dependence in `hybrid.py` is intended is a separate
   ANYmesher question; the route reproduces the whole-model result exactly.
7. `NativeProjectMeshingSession` (incremental, per-component, background threads)
   is a different path: it meshes one sheet/face/member per request. Out of scope
   here; it already splits by component, though its threads share the GIL.

## 3. Where the pool lives: options

| | Description | Warm pool | GUI responsiveness | Cancel / cleanup | Change size |
|---|---|---|---|---|---|
| P1 | Headless and `thread`-executor callers only (scripts, batch, tests, CLI, `ANYFEM_JOB_EXECUTOR=thread`). Pool owned by the calling process. GUI process executor stays serial. | yes (module-level, caller supplied) | n/a (no GUI) or unchanged | simple (same process tree, caller owns it) | small |
| P2 | GUI: mesh worker made non-daemonic, one pool per heavy job, tree-killed through a Windows Job Object assigned at worker start (kill-on-close handles GUI crash; `TerminateJobObject` handles cancel). | no (cold start 1-3 s per job, so only for big jobs) | worker side stays off the GUI interpreter | needs the Job Object; the existing parent-watch thread stays | medium (`job_process.py`, small ctypes block) |
| P3 | GUI process is a broker: it owns a warm pool; the mesh worker returns the prepared component payloads, pool processes mesh them and one pool task joins and finishes (remap) | yes | broker only forwards bytes; no heavy work in GUI | pool recreated when broken; cancel kills pool processes | large (splits `Project.generate_mesh` into stages) |

**Recommendation:** P1 first (it delivers the speed-up to scripts and batch runs
with almost no risk and lets the policy be tuned), then P2 for the GUI behind
the same policy. P3 only if the measured cold start (1-3 s per heavy job)
matters in practice. P2's trade-off: no warm pool.

## 4. ANYmesher prerequisites (small; I can do these)

1. **Cancellation that works.** Wait with a short timeout, call
   `cancellation_check` each tick, and on cancellation cancel pending futures and
   terminate worker processes; document that a reused executor is then broken and
   must be recreated by its owner. (Today: checked only between completions.)
2. **Hash-safe diagnostics.** Move machine/run dependent values (`workers`, wall
   and per-component timings) under one key that ANYfem can drop, or make the
   deterministic remainder (`used`, `components`, `separation`, `merge_reasons`)
   the only hashed part.
3. **Work estimate.** Expose a cheap estimate of elements per component
   (face area over target size squared, or `Seeding.total_elements_estimate` when
   a seeding exists) so callers can apply a threshold; add
   `ParallelOptions.min_estimated_elements` defaulting to the cold-start
   break-even below.
4. **Mesh-returning wrapper** `generate_hybrid_mesh_parallel` (ANYfem calls the
   `Mesh`-returning `generate_hybrid_mesh`).
5. **Failure policy.** Today a dead worker raises `MeshError`. Decide whether a
   job should fall back to the serial route once (recording it) instead.
6. **Progress.** Report "component k of N" through the existing
   `cancellation_check` stage argument.

## 5. ANYfem-side changes (file level, for its owner)

- `model/project.py`: replace the single `generate_hybrid_mesh(working,
  **meshing_parameters)` call by a small helper that chooses the route by policy
  (section 6) and otherwise calls it unchanged. The `automation` branch
  (`generate_automatic_mesh_result`) stays serial. Use `.mesh` of the parallel
  result.
- Setting: a mesh execution policy (`off` | `auto` | worker count), default
  `off` until validated, then `auto`. It is an execution policy, not mesh input:
  keep it out of `mesh_input_hash` (the mesh is equivalent up to numbering).
  Optional environment override `ANYFEM_MESH_WORKERS` for headless use.
- `mesh_jobs.py`: `mesh_semantic_hash` drops the runtime block from section 4.2.
  Record the route and fallback reason in the mesh job log/diagnostics so users
  can see why a job ran serial.
- P2 only: `job_process.py` (non-daemon worker + Job Object, parent-watch retained)
  and `mesh_jobs.py` (pass executor/pool policy into the worker).
- GUI (later): a mesh dialog option to use several processes, off by default;
  show the reason when it falls back.

## 6. Policy for `auto`

Use the parallel route only if all hold, otherwise serial:

- not the mapped fast path, no `automation`, no `quad_first`, no `strict`
  certification, no change set (the route would refuse anyway; test first to
  avoid a plan call);
- at least 4 independent components (plan result), and estimated total elements
  at least about 3,000 with a cold pool (about 500 with a warm one). Basis:
  measured about 1.5 ms per native plate element and cold start 1.2 s (4-16
  workers) to 3 s (32 workers). These numbers are from one machine and must be
  re-measured before adoption;
- workers = `min(components, cpu_count - 2, 16)`, never more than 61 on Windows,
  at below-normal priority (ANYfem already has a priority helper) so the GUI wins
  contention. Memory budget: each worker is roughly 70-130 MB resident before
  mesh data (observed process working sets), so 16 workers is about 1-2 GB.

## 7. Verification plan (acceptance)

1. **Parity through ANYfem:** port `exp8_anyfem_project.py` into a test: serial vs
   routed `Project.generate_mesh` are equal up to numbering for plates, a model
   with beams (member ids, `beam_offsets`, couplings), quadratic order,
   refinements, and a precomputed seeding.
2. **Fallbacks:** each refusal in section 1 yields the serial mesh and a recorded
   reason; `strict` certification unchanged.
3. **Hash:** same mesh_hash for 4, 8 and 16 workers on one machine; solve jobs
   still bind to it.
4. **Cancellation:** cancel during a multi-component job returns within a stated
   bound (target under 1 s) and leaves no child processes (check the process
   list), including after a simulated GUI crash (P2).
5. **Failure injection:** kill one pool worker mid-job; the job fails cleanly (or
   falls back, per 4.5) and the pool is recreated.
6. **GUI responsiveness:** reuse the tick-timing harness from the worker-process
   work; p99 stall during a routed job must not exceed the serial worker-process
   baseline (P2/P3).
7. **Benchmarks:** table of serial vs routed for 700-element and 70-element
   components at 4, 16, 32 components, cold and warm, with the threshold
   derived from it.
8. Run ANYfem's meshing tests with and without the route enabled; failure sets
   must match the baseline (two pre-existing failures today, unrelated).

## 8. Phasing

1. ANYmesher prerequisites 4.1-4.4 (small). Decision needed: 4.5.
2. P1 in ANYfem behind `off` by default, with tests from section 7.1-7.5. Gives
   scripts/batch the speed-up.
3. Calibrate the policy thresholds on the target machines; switch default to `auto`
   for P1 callers.
4. P2 for the GUI (Job Object), then 7.4-7.6.
5. Revisit P3 only if cold start proves to matter.

## 9. Open questions for the owner

- Is the GUI the main target, or scripted/batch meshing? That decides whether P2
  is worth doing early.
- Is it acceptable that parallel and serial meshes differ in numbering (same
  mesh, different `mesh_hash` for the same project)?
- Failure policy: raise or fall back to serial?
- Is the face-count dependence of the planar chart metric in `hybrid.py` intended
  (single plate 32 elements, same plate in a 2-plate model 73)? The route
  reproduces it exactly; fixing it would change serial output.
- Not supported by the route today: `strict` certification, `quad_first`,
  automation recovery, partial face/member selections. Are any of these needed
  for the models that benefit most?
