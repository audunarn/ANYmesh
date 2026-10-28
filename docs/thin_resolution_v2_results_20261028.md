# Thin-region candidate v2 — measured NO_GO

The authorized h=0.002 m (thickness/5) candidate completed both end-to-end cases. It is **NO_GO**: both orientations pass every geometric predicate, but the aligned case fails three unchanged physical predicates for each of P1 and P2. The rotated negative-offset case passes all geometric and P1/P2 predicates. Independent final review is pending; this author report does not establish acceptance or provider readiness.

Only sizing metadata changed from the original h=0.0025 m study. Original geometry.json, snapshot.json and input.npz were copied byte for byte. All original source model/region/face IDs, coordinates, signatures, revisions, face groups, group owner semantics and region groups match. All 28 recorded semantic comparisons pass. Original failed evidence and exhausted authority remain intact.

| Case | Original nodes / cells | V2 nodes / cells | Max axial span / thickness, old → new | V2 max regular-tet condition | P1 / P2 | Verdict |
|---|---:|---:|---:|---:|---|---|
| thin_sampling_aligned | 2897 / 12940 | 6147 / 29440 | 0.497543093157 → 0.348846866853 | 5.02985874514 | FAIL / FAIL | NO_GO |
| thin_sampling_rotated_negative_offset | 2631 / 11670 | 6562 / 32313 | 0.502124028685 → 0.404339773687 | 4.57945231861 | PASS / PASS | PASS |

There were 20/20 passing geometric predicates, 38/44 passing physical predicates, and 6 failed physical predicates. Both orders ran for both cases. Coefficient and cell caps (150000 / 100000) were respected. No geometric failure was followed by physics.

Aligned failing comparisons (limits: resultant 1e-8, residual 1e-9, energy/work 1e-10):

| Order | Face force/moment error | Affine residual | Energy error | Work error |
|---|---:|---:|---:|---:|
| P1 | 5.65682846459e-08 | 2.98296112125e-08 | 8.08592570678e-09 | 3.73648174179e-07 |
| P2 | 5.6568284567e-08 | 2.67018730796e-08 | 8.08592337531e-09 | 3.73648172292e-07 |

The failing predicate names are face_force_moment, affine_equilibrium and positive_energy_and_work. Aligned geometric relative volume error was 8.08592548474e-09, maximum face-area error 2.26622056465e-06, and maximum plane distance 4.14183786272e-07 m; each remains inside its unchanged geometric limit. A cause of the physical mismatch has not been established. No tolerance was relaxed and no corrective remeshing occurred.

| Case | Supervised complete-case seconds | Native-child process seconds | Native meshing seconds | Native-child lifetime peak RSS bytes | P1 / P2 coefficients |
|---|---:|---:|---:|---:|---:|
| thin_sampling_aligned | 405.443420885000 | 2.591837009000 | 2.087019407000 | 95985664 | 18441 / 131388 |
| thin_sampling_rotated_negative_offset | 448.507262533000 | 7.455787228000 | 6.415012900000 | 140554240 | 19686 / 142467 |

RSS is Linux RUSAGE_CHILDREN for the isolated native child after reaping, including Python/NumPy/provider imports, not full case or FE memory. Both native receipts report zero owned processes after cleanup. All seven outer supervisor receipts verify cleanup with zero remaining owned members. No case or native timeout occurred.

Exactly 2/2 native call claims were consumed, one per orientation, with no retries. All process attempts were charged: 939.985701038000 / 1200 seconds used, 260.014298962000 seconds preserved. The 90-second reconciliation reserve was retained. The first control attempt failed because the unchanged exact native runtime lacks pytest (0.06680005 seconds); that evidence remains. The existing pytest-capable linux-runtime then ran only the changed registration/reuse tests: **14 passed, 0 failed**. No installation or runtime modification was made. Existing original 14-test controlled metric/supervisor evidence was reused after unchanged metric AST and runner/worker/supervisor/physics hashes were checked. Fresh freeze and fresh-process verification passed before native release. The final reconciliation exits 1 because its measured aggregate verdict is NO_GO.

Exact supervised commands, in execution order (all under the unique evidence directory, commands and effective environments also in planned-processes and processes receipts):

```text
/mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B -m pytest -q tests/test_thin_volume_v2_qualification.py --junitxml=/mnt/c/ANYworkspaceWork/workflow2-20261027/evidence/thin-resolution-v2-20261028-a1/linux-unit.xml --basetemp=/tmp/anymesh-thin-v2-study-units
/mnt/c/ANYworkspaceWork/workflow2-20261027/linux-runtime/bin/python -B -m pytest -q tests/test_thin_volume_v2_qualification.py --junitxml=/mnt/c/ANYworkspaceWork/workflow2-20261027/evidence/thin-resolution-v2-20261028-a1/linux-unit.xml --basetemp=/tmp/anymesh-thin-v2-study-units
/mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_thin_volume_v2.py freeze
/mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_thin_volume_v2.py verify
/mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_thin_volume_v2.py case --case thin_sampling_aligned
/mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_thin_volume_v2.py case --case thin_sampling_rotated_negative_offset
/mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/../evidence/thin-resolution-v2-20261028-a1/reconcile.py
```

Frozen study SHA256: e4a362d60012cb31be690ab2a91e3be52db492ac2ff76580e6f001f5ea0b8452. Runtime SHA256: 3d0269bc260c0a829d4541f33da13ae513596a46f9118002e1ef9cc767bc99a8. Reuse-proof SHA256: b57a875c05f615c45736daac5fc1ecb24d461ad463231e95c4f25cf00e1b3200. The existing pytetwild 0.4.2+anyworkflow2.exact1 provider and installed ANYsolid were used unchanged; native and shared-library/source identities were frozen and freshly verified. No native rebuild, import patch, broad regression suite, primary-checkout edits, merge/push or publication was performed.

Evidence: ../evidence/thin-resolution-v2-20261028-a1/terminal-summary.json, budget.json, frozen/reuse-proof.json, root-prospective-review.json, internal-release.json, linux-unit.xml, cases/*/result.json, cases/*/geometry-checks.json, cases/*/order*-checks.json, native-attempts/*, planned-processes/*, processes/*/result.json and full stdout/stderr. Review status: prospective root non-author review accepted the frozen execution candidate; independent final non-author same-family Read review supports terminal NO_GO (workspace evidence/thin-resolution-v2-20261028-a1/independent-final-review.json). No further native execution remains authorized. No general FE accuracy, performance improvement, complete Workflow 2 readiness or provider promotion is claimed.
