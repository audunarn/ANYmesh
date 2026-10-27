# Workflow 2 volume implementation

State: verified against assigned controlled-development criteria; Level A,
not qualification. Real provider UNQUALIFIED / blocked and public dispatch
explicitly disabled pending a root-owned capability decision.
Owner: mesher worker; isolated ANYmesh only, baseline 0415256e.

Plan and acceptance:
1. Immutable tet4 sibling and source snapshot preflight. Accept controlled convex
   normal-prism fixtures; refuse malformed, unsupported and ambiguous inputs.
2. Independent output admission: positive Jacobians, complete outward boundary,
   connected manifold cells, pair intersection conformity, convex-plane/loop
   ownership, per-face coverage, protected source edges/corners and volume.
   Acceptance: positive fixture plus targeted negative mutations.
3. Lazy pinned pytetwild subprocess adapter. Accept only complete validated raw
   arrays; bound timeout/cancellation, terminate owned children, preserve effective
   options and provenance. Acceptance: fake-worker process failure matrix; real
   provider only after root W2.0 disposition.
4. Public exports, optional extra, docs and advisory focused profiles. Acceptance:
   focused source tests; root owns installed consumer and independent review.

Dependencies: shared ANYgeometry solid_volume_contract.md; root owns W2.0 real
provider feasibility and final acceptance. Risks: provider boundary error may not
meet planar protected-feature admission; exhaustive intersection checks prioritize
correctness over an unqualified scalability claim. No meshing algorithm added.

Implemented files:
- `src/anymesher/volume.py`, `_volume_validation.py`, `_volume_process.py`,
  `_volume_worker.py`: immutable complete structural protocol, independent
  source/numerical/geometry admission, canonical rows and recoverable provider
  maps, lazy pinned adapter, typed atomic refusals, retained process evidence.
- `src/anymesher/__init__.py`, `pyproject.toml`: public exports and optional extra.
- `tests/test_volume_mesh.py`, `tests/test_volume_process.py`: controlled geometry,
  corruption, stale identity, mutation, fake-provider protocol and process tests.
- `tests/test_layering.py`: scoped optional worker import; ANYsolid forbidden as
  a core dependency. `THIRD_PARTY_NOTICES.md`: separate MPL-2.0 provider notice.
- `tools/run_dev_smoke.py`, `tools/run_affected_tests.py`: explicit volume profile
  and child-worker dynamic mapping. Frozen SG1 guard moved ahead of file existence
  check so missing SG1 paths still receive the correct scientific-gate refusal.
- `docs/volume_mesh.md`: interface, blocked capability and honest validator scope.

Initial isolated checkout clean. Root-owned concurrent feasibility, check-runner,
consumer test, CI and handoff files were not edited by this worker. No sibling
writes, environment mutation, real mesher execution, commits or publication.

## Verification evidence

All paths below are under
`C:/ANYworkspaceWork/workflow2-20261027/evidence/`. Each contains the exact command,
source SHA-256 identities, full stdout/stderr, terminal process-wall accounting
and cleanup receipt. Pytest commands also contain JUnit except the profile runner
(its console output retains the count). Process limit 180s per command, cumulative
600s authorized. Actual cumulative worker checks: **117.641s**, **482.359s remaining**.

| Evidence directory | Result | Process-wall seconds |
| --- | --- | ---: |
| mesher-initial-751da3d807 | 47 passed, 2 failed | 3.922 |
| mesher-thin-repair-5cc52efb1b | 48 passed, 1 failed | 2.625 |
| mesher-exact-contact-86c92aec1b | 49 passed | 2.422 |
| mesher-adversarial-aec3f783e3 | 57 passed | 2.406 |
| mesher-profile-7820b0a426 | 60 passed | 6.829 |
| mesher-owner-layering-32f7ec7bc0 | 70 passed, 1 failed | 21.688 |
| mesher-final-source-4d52ffec60 | 73 passed | 23.562 |
| mesher-final-typed-46950c6143 | 78 passed | 24.078 |
| mesher-guarded-source-090f1ac709 | 80 passed | 24.984 |
| mesher-canonical-2f559c2542 | 70 passed | 5.125 |

Failure history retained: initial rotated thin-prism clipping was ill conditioned;
resolved by exact rational contact predicates on unchanged binary coordinates
instead of tolerance relaxation. Oblique fixture initially also admitted a valid
alternative normal-prism direction; corrected fixture to shear both cap axes.
Existing affected-runner SG1 guard-order failure repaired as described above.
The last canonical-row-only change was verified by the 70-test focused run;
the 80-test aggregate already passed all unchanged layering/metadata/profile code.

Runtime executable for all checks:
`C:/ANYworkspaceWork/workflow2-20261027/runtime/Scripts/python.exe`.
Harness: `tools/workflow2_check.py --name <evidence-prefix> --cwd
C:/ANYworkspaceWork/workflow2-20261027/ANYmesh --evidence-root
C:/ANYworkspaceWork/workflow2-20261027/evidence --pythonpath
C:/ANYworkspaceWork/workflow2-20261027/ANYgeometry/src --pythonpath
C:/ANYworkspaceWork/workflow2-20261027/ANYmesh/src --source <recorded sources>
--timeout 180 -- <command>`. Exact expanded arrays are in each `identity.json`.

Executed pytest command suffixes:
- `python -m pytest -q tests/test_volume_mesh.py tests/test_volume_process.py`
  (initial, thin-repair, exact-contact, adversarial and canonical runs).
- `python tools/run_dev_smoke.py --profile volume` (profile run).
- `python -m pytest -q tests/test_volume_mesh.py tests/test_volume_process.py
  tests/test_layering.py tests/test_affected_test_runner.py
  tests/test_packaging.py::test_allowed_third_party_imports_are_declared_dependencies
  tests/test_packaging.py::test_version_matches_pyproject`
  (owner-layering, final-source, final-typed and guarded-source runs).
- `git diff --check`: passed after final implementation.

## Handoff and remaining gates

Non-author review: root-owned, under review; worker findings are not acceptance.
Root reported controlled installed ANYsolid P1/P2 consumer checks separately;
those are not claimed as this worker's test runs. No real provider or numerical
qualification ran here. Root W2.0 feasibility has preserved failed import and
timeout evidence; the import-only local wheel patch alone is not capability.
The normal public provider route refuses before launching. Fake-provider tests
alone temporarily enable the private seam after installing a controlled launcher
or distribution. No public API/environment bypass was added.

Windows owned-tree termination, zero-active Job Object count, cancellation,
timeout, logs and orphan-child cleanup were exercised. POSIX group cleanup now
checks the process table for active owned members after kill/reap; Linux procfs
parsing is controlled-fixture tested. Native POSIX process execution was not run
on this Windows worker and remains a root/reviewer verification item.

Known limits: convex normal prisms only, first-order tet4, source-feature-resolved
boundary tolerance, potentially quadratic candidate intersections in worst-case
AABB overlap. No scalability claim. Independent admission uses floating-point
roundoff predicates with exact resolution of ambiguous thin contact, and bounded
geometric boundary strips; it is not qualification for arbitrary geometry.
Installed wheel/cross-repository integration, real-provider capability disposition,
independent acceptance, full CI/release gates and any delivery remain root-owned.
