# Dynamic and Incremental Remeshing

Date: 2026-08-12

## Implemented correctness contracts

- `GeometryMeshingView` binds all derived indexes to one ANYgeometry model ID and revision and rejects stale model access.
- `ChangeSetQueue` requires contiguous revisions and coalesces revisioned changes without inventing geometry state.
- `StructuralMeshingPipeline.affected_components` maps public `ChangeSet` records to structural components using persistent Sheet, Member, Attachment, Junction, face, and edge ownership.
- `ComponentGenerationCache` uses per-component generation tokens so invalidated background jobs cannot publish stale results.
- Unaffected component generations may remain publishable across a later model revision, while affected components are invalidated explicitly.
- ANYfem's headless `NativeMeshingRuntime` regenerates only resolved dirty components, retains the exact clean publication, and performs zero generation for an empty dirty resolution.
- Serial and two-worker component scheduling publish byte-identical canonical mesh payloads for the same authoritative faces, including reversed submission order.
- Native hybrid and planar-surface generation accept an optional safe-phase `cancellation_check(stage)` callback. ANYfem passes its existing cancellation token, so an in-flight cancellation is observed after triangulation and before recombination or publication.
- `Project.generate_mesh` and the desktop `MeshTaskManager` forward the same callback, exposing safe-phase cancellation to the app-facing job bridge without UI-thread calls.
- Local element activity updates exposed sides and reuses the mesh BVH without rebuilding geometry truth.

Changed-region audit remains deliberately non-certifying. Interactive paths may use `audit_changed_region`, affected AABBs, and the maintained ANYgeometry index, but only a full strict audit can support certification.

## Focused evidence

```powershell
$env:PYTHONPATH='C:\Github\ANYmesh\src;C:\Github\ANYgeometry\src'
python -m pytest tests\test_structural_pipeline.py -q
```

Result: `9 passed in 0.09s`.

The focused cases cover:

- Sheet/member reverse indexes and stale-view rejection.
- Contiguous `ChangeSet` coalescing.
- Invalidated-job rejection and unaffected-cache publication.
- Exact edge parameters and multi-edge Member stationing.
- T3, T6, Q4, and Q8 inverse interpolation through the mesh BVH.
- Local declared-only structural preflight and coupling.
- Local element deactivation/reactivation, exposed-side updates, and BVH synchronization.

The runtime publication slice was separately and independently reproduced as `11/11` passing in its declared scope.

```powershell
$env:PYTHONPATH='C:\Github\ANYfem\src;C:\Github\ANYsolver\src;C:\Github\ANYmesh\src;C:\Github\ANYgeometry\src;C:\Github\ANYio\src;C:\Github\ANYmaterial\src;C:\Github\ANYbuckling\src'
python -m pytest tests\test_native_meshing_runtime.py tests\test_native_meshing_cache_reuse.py -q
```

Result: `7 passed in 0.96s`. This adds direct clean-publication reuse, dirty-only regeneration, and empty-resolution zero-generation evidence to the existing runtime behavior.

A focused real-meshing comparison in `tests/test_native_meshing_cache_reuse.py` additionally completed `2 passed in 0.90s`: one-worker serial generation and two-worker concurrent generation produced identical canonical persisted mesh payloads per component despite reversed submission order.

Safe-phase cancellation evidence:

```powershell
python -m pytest tests\test_native_cancellation.py -q
```

Result in ANYmesh: `1 passed in 0.13s`.

```powershell
python -m pytest tests\test_native_meshing_backend.py -q
```

Result in ANYfem: `3 passed in 0.98s`. The in-flight case blocks a real native triangulation, requests cancellation, then proves recombination never starts and the exact previous publication remains current.

The app-facing bridge was then qualified headlessly with `tests/test_native_meshing_backend.py` and `tests/test_mesh_task_safe_cancellation.py`: `4 passed in 0.98s`. Its canceled job emits `cancelling` followed by `cancelled`, never emits `completed` or `failed`, and never starts recombination.

## Open acceptance gates

- Safe-phase latency inside very large triangulation or mapped-kernel calls; cancellation is cooperative and never kills a worker thread.
- Parallelism inside a single native component; the qualified result covers deterministic concurrent component scheduling.
- Leased scaling and peak-memory measurements at the governing plan sizes.
- Full application/UI integration and long-running edit/restart regression.

## Leased incremental acceptance result

The production ANYfem component runtime was qualified on Windows 11 with
CPython 3.13.9 using 20 persistent Sheet components and two bounded workers.
The exact report is `incremental_performance_results.json`.

- Initial full generation: 8,000 elements in `37.870554 s`.
- Dirty response: one 400-element component in `2.045820 s`.
- Dirty scope: `1/20`, or `0.05`.
- Full-over-dirty speedup: `18.5112x`.
- Clean publication object reuse: passed.
- Clean canonical hash stability: passed.
- Dirty publication replacement: passed.
- Governing `dirty <= 0.20` and `speedup >= 5x` gates: passed.

This result is scoped to the recorded model, machine, interpreter, and component
scheduler. It is not a broad generation-scaling, peak-memory, or cross-platform
performance claim.
