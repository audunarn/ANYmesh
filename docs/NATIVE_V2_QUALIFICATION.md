# Native-v2 qualification

Native-v2 is opt-in in 0.5.0. The accepted legacy reference is commit
`27e428188a891705288fef82bab0b166e330aff2`; default options must preserve its
canonical points, topology, associations, quality decisions, serialization,
and cancellation boundaries, except independently reviewed correctness fixes.

The corrected implementation baseline is
`246658c17096cd6e07075c4a4596c28e10c318a6`. Its source dependency and local test
identities are recorded in `native_v2_corrected_baseline.json`. The historical
reference remains retained; this does not supply missing platform-scale evidence.
The historical expanded roadmap is `FIELD_GUIDED_0_5_IMPLEMENTATION_PLAN.md`.
The user's finalization decision supersedes its quad-first release scope:
field-guided orientation solving and advancing quad fronts are deferred to a
later release. They are not implemented and are not advertised as 0.5.0 features.
Current closeout covers native-v2 and cylindrical Frontal-Delaunay using
owner-certified connected sector faces. True single-face periodic topology
remains deferred. This scope decision does not waive the remaining correctness,
performance, source, installed-wheel, or hosted-CI qualification gates.

The evidence runner is `benchmarks/native_v2_baseline.py`. CI runs
`check-evidence`, which first validates the immutable contract and then executes
the fixed bounded native test-node corpus. A `run` requires an exact source commit, install kind,
compiler identity, backend, case and scale; wheel runs additionally require the
wheel SHA-256. It performs exactly one warmup and seven measurements and records
all repetition hashes, topology and Q4 distribution, alignment, serialization
size, quality, cancellation, actual backend, environment, and peak RSS. Windows
and Linux retain paired source and installed-wheel evidence. macOS retains
normal build/smoke coverage. The `500k` and workstation scales additionally
require `--allow-large` and are never CI.

The registered scales and labels can be inspected without running heavy cases:

```text
python benchmarks/native_v2_baseline.py check-contract
```

The CI evidence gate builds the extension and executes the bounded corpus:

```text
python benchmarks/native_v2_baseline.py check-evidence
```

The command first resolves every declared corpus path. Frozen performance
records may be supplied with repeatable
`--evidence-pair LEGACY_JSON FRONTAL_JSON` arguments; every pair is consumed by
the acceptance comparator for backend, environment, target, configuration,
quality, protected topology, associations, ownership, activity, mapped
zero-use runtime/RSS, frontal runtime, and native-v2 RSS.

This bounded gate does not replace platform-scale qualification. Merge remains
blocked until reviewed Windows and Linux source and installed-wheel records
cover the registered 10k, 100k, and 500k scales, including mapped zero-use.
When no pairs are supplied the command reports `not_supplied_merge_blocker`;
missing evidence is never inferred or fabricated.

Each legacy/frontal pair is accepted with the explicit comparator:

```text
python benchmarks/native_v2_baseline.py compare --legacy LEGACY.json --frontal FRONTAL.json
```

Acceptance first requires exact backend, environment, target/configuration,
quality-policy, protected-topology, association, ownership, and activity
contracts. It then rejects any measured quality regression before comparing
performance: 100k Frontal median at most 1.25 times legacy, mapped zero-use median
regression at most 3%, mapped peak-RSS regression at most 5%, and native-v2
peak RSS at most twice legacy. Evidence directories are immutable and named by
source commit and configuration digest.

Compiled metric and topology kernels check Python signals at least every 4096
records throughout material native loops.
Application cancellation remains governed by `cancellation_interval`: the
orchestrator checks before the initial queue and after each configured block of
topology operations, while each native insertion is one bounded linear scan of
the current topology. This describes the frozen implementation, not the final
incremental-foundation target: persistent local topology, certified location
and removal of routine full-mesh per-edit scans remain required.
Near-zero orientation or in-circle predicates decline
native dispatch and use the repository adaptive Python predicate rather than
making a platform-dependent `long double` decision.

The corpus covers planar and curved legacy paths, intersections, declared
junctions, holes, narrow ligaments, mixed mapped/native components, activity,
incremental components, cancellation, and deterministic transforms. Cylindrical
Frontal-Delaunay qualification must additionally establish work-bearing native
refinement, physical metric sizing, exact sector/shared-node conformity, and
protected trim ownership; mapped cylinder walls are not evidence of that route.
Other curved Frontal-Delaunay surface types, global matching, multiblock
expansion, adaptive remeshing, and field-guided quad fronts remain deferred.
Existing collars and triangle recombination are not the deferred quad-first
architecture. Legacy behavior remains the default.

## Finalization checkpoint: 2026-09-08

The following are observed closeout facts, not a release approval:

| Gate | Current evidence and remaining work |
| --- | --- |
| Release scope | Native-v2 and owner-certified cylindrical sector refinement; quad-first deferred by user decision. |
| Source identity | ANYmesher HEAD `76b9e7af534fb6c44a1259afc9e3d923bf453606` has uncommitted implementation and test changes. It does not identify the current qualified source by itself. |
| Geometry identity | ANYgeometry HEAD `b147a8be413ca7e702697b2bc2d75560d4c27806` has uncommitted cylinder-patch changes, including the orthogonal helper. Its HEAD alone is not the tested dependency identity. |
| Focused checks | The latest local export correction passed 299 tests; retained evidence is `reports/local_export_delta_561c_20260908`. This is bounded source evidence only. |
| Full source suite | Started under `reports/final_source_suite_a782_20260908`; completion requires its terminal exit and JUnit records, not this checkpoint. |
| Native origin | That suite explicitly preloads the ABI-2 artifact in `reports/native_capsule_abi_8e17_20260908/build-lib/anymesher/`; it does not qualify the stale editable native binary or subprocess import origins. |
| Package metadata | The suite reported editable metadata ANYmesher 0.4.0 and ANYgeometry 0.4.2. These labels must not be mistaken for the current source version or a release-wheel identity. |
| CI dependency | The retained Tests workflow pins ANYgeometry `dd954f088a4cb95e267280cc4777b09e16232bd9`. Bind an available, reviewed successor containing the required owner changes before claiming hosted qualification. |
| Platform qualification | Current Windows/Linux source and installed-wheel evidence, macOS smoke, and the normal CI matrix remain required. Older wheel results do not cover the latest storage/export changes. |
| Performance qualification | Profiling and optimization are stopped for finalization. Existing matched-count, mapped-regression, scaling and memory acceptance gates are not waived; incomplete or unmatched measurements are not passing evidence. |

Preserve historical failures, binaries and unrelated worktrees. Do not refresh
the dependency environment or replace a loaded native binary during the active
source suite. Freeze the final deliberate source and dependency pair before
the installed-wheel and hosted qualification handoff. No upload, tag or release
publication is implied by this checklist.

## Completed Windows checks and remaining handoff

The following completed runs supersede the pending local-run status above,
without rewriting the original failure evidence:

| Check | Terminal result | Evidence under `reports/` |
| --- | --- | --- |
| Corrected complete source suite | 1200 passed, 29 skipped, no failures | `final_source_suite_corrected_ae31_20260908/` |
| Complete installed-wheel suite | 1200 passed, 29 skipped, no failures | `final_wheel_pair_c47e_20260908/full-installed-suite/` |
| Owner complete source suite | 878 passed | `owner_full_closeout_287e_20260908/` |
| Dedicated disabled-native packaging/triangulation | 19 passed, 4 compiled-only skips | `disabled_native_closeout_89a4_20260908/` |
| Explicit Windows build-supervisor checks | 9 passed after installing the missing test-only psutil dependency; original setup failure retained | `build_supervisor_closeout_ready_f83d_20260908/` |
| Optional Gmsh/backend/ownership contracts | 34 passed, including cases previously hidden by module-level skips | `gmsh_closeout_c23d_20260908/` |
| Clean runtime installation | Offline install from resolved wheels, pip check, isolated imports and required-native wheel smoke passed | `clean_install_closeout_764e_20260908/` |

The clean environment does not inherit system site-packages. Its frozen runtime
is ANYmesher 0.5.0, ANYgeometry 0.4.3, NumPy 2.4.6 and Shapely 2.1.2. Native
and owner wheel identities, strict Twine results and archive RECORD verification
are retained in `final_wheel_pair_c47e_20260908/`. The complete source run used
source Python modules with that qualified native artifact explicitly preloaded;
it does not prove editable-DLL replacement. Optional GUI checks remain opt-in.

Both configured checkouts are on `main`; their remotes are
`https://github.com/audunarn/ANYmesh.git` and
`https://github.com/audunarn/ANYgeometry.git`. This read-only configuration check
does not establish remote synchronization or a final delivery commit.

Remaining release-qualification gates:

1. The user-approved orthogonal multi-hole correction is complete: each hole
   is checked against outer-loop winding, while pairwise clearance and explicit
   nested-hole rejection remain enforced. Eight new focused cases passed;
   the owner contract selection passed 108 tests and the complete owner suite
   passed 886. The new owner wheel also passed 36 installed-wheel mesher binding
   and Frontal-Delaunay integration checks. Evidence is retained under
   `owner_multi_hole_closeout_09df/`, `owner_multi_hole_full_0b76/` and
   `owner_multi_hole_wheel_8aa0/`. Earlier complete mesher runs remain evidence
   for their earlier dependency pair, not a relabeled complete rerun with this
   corrected owner wheel.
2. Freeze and review the deliberate ANYmesher/ANYgeometry source pair, then
   bind available immutable owner commits in the workflows and matching guards.
   Preserve unrelated changes, worktrees, old binaries and failure evidence.
3. Complete hosted CI and the remaining Linux/source/installed-wheel and macOS
   smoke lanes for that pair. Windows results are not cross-platform evidence.
4. Complete the registered matched-count 10k/100k/500k and workstation evidence,
   runtime/memory comparisons and mapped zero-use gates. Existing exploratory
   profiles with mismatched element counts are not accepted comparator pairs.
   Stopping optimization does not waive qualification thresholds.

No upload, tag, GitHub release, automatic-default change or quad-first
implementation is part of this handoff. The release remains unapproved until
its outstanding gates are satisfied or explicitly superseded by the user.
