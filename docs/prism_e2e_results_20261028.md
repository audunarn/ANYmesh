# Retained pentagonal-prism E2E — finite PASS

The single root-reviewed retained pentagonal-prism case **passed all 9 geometric predicates and 22 P1/P2 physical checks**, followed by complete production readmission. Zero native meshing calls and no physics retry. This is candidate-source ANYgeometry/ANYmesher plus installed ANYsolid, one retained resolution; not new installed-owner/platform/native-core coverage or provider promotion.

Root independently inspected the authored harness and reconciled frozen hashes,
the complete check inventory, budgets and supervised cleanup. Disposition:
**accepted finite retained-prism E2E only**. Review is retained at
`evidence/prism-e2e-20261028-a1/root-final-review.json`; no different-family or
human final workflow acceptance is claimed.

Frozen identity: `4813fb9638d042104cc6db98a765bff79964c3e590c84bc07f2b35b5cb9c574f` at `evidence/prism-e2e-20261028-a1/frozen/identity.json` in the workspace. Registration/refusal controls: 7 passed / 0 failed. Freeze and fresh-process verification passed with descendant cleanup verified.

| Result | P1 | P2 |
|---|---:|---:|
| Physical checks | 11/11 PASS | 11/11 PASS |
| Coefficients / fixed / free | 1389 / 237 / 1152 | 9219 / 855 / 8364 |
| Prescribed tension sparse LU solves | 1 | 1 |
| Affine equilibrium residual | 7.765242333895177e-14 | 2.0145355498242336e-13 |
| Energy error | 2.220446049250313e-16 | 4.440892098500626e-16 |
| Solved displacement error | 1.085217603650415e-15 | 2.9742827642445984e-15 |

Full checks include stiffness symmetry, independent physical rigid rank/span/residual, face force/moment, affine equilibrium, full tensor recovery, positive energy/work, body and pressure resultants, prescribed-tension displacement, free equilibrium and reaction balance. Both recovered mean tensors, all per-face force/moment values, analytical references and solve counts are saved in `case/order{1,2}-diagnostics.json`; unchanged helper journals and solved coefficient fields are retained. Analytical positive strain work is 0.01843586477741374 for the specified affine patch, not a general accuracy claim.

Maximum regular-tet condition 4.624280501975736 < 32; plane deviation 1.5871787729456253e-17m < 6e-6m; relative volume and maximum face area errors 2.220446049250313e-16. Readmission verifies 498 exterior facets, 7 source faces, complete unique source ownership and exact overlapping groups. Source and admitted mesh volume are both 0.0014265847744427303m³. The source owner, face identities and selections remain those of the original frozen pentagon; no meshing/refinement-pair claim is made.

The regular pentagon has radius .1m, height .06m, 463 retained nodes and 1899 real Tet4 cells. Exact original owner geometry is loaded with `anygeometry.from_dict`; raw SHA256 `5e2b1437a16ec1f84058617dfb2de7ef25996a1990eaaffd936cc4d73e00ef67`. No box reconstruction occurs. Five installed ANYsolid package files match registered wheel SHA256 `601bd500625e6cd95f3a741f4f19b87bdcb4a72a7c7650278fea8be3efc56c3c`.

Exact Windows launch prefix (workdir the candidate ANYmesh):

```powershell
wsl.exe -d Ubuntu-24.04 --exec /mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_retained_prism_e2e.py run-controls
wsl.exe -d Ubuntu-24.04 --exec /mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_retained_prism_e2e.py run-freeze
wsl.exe -d Ubuntu-24.04 --exec /mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_retained_prism_e2e.py run-verify
wsl.exe -d Ubuntu-24.04 --exec /mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/qualify_retained_prism_e2e.py run-case
```

Terminal budget: 119.60470601699993/600 seconds charged, 480.3952939830001 remaining. Supervised process walls: controls 1.9285781549999683s, freeze 29.88479542199991s, verify 28.92672781600004s, case 55.514749424s. Every owned process cleanup is verified with zero remaining descendants. Each process is capped at 300s; case execution retained 45s reserve. First sandbox WSL access denial and wrong-default-distro startup failure are distinct retained pre-supervisor failures, conservatively charged by tool wall (3.3498552s), not reconstructed subprocess wall. Controls actually passed but post-run duplicate-key bookkeeping failed; the exact controller receipt was recovered without rerunning controls. The repaired bookkeeping was exercised by successful freeze/verify/case. Full logs, runtime/source hashes, exact effective environments and cleanup are under the evidence root. `launch-failures.json` and `bookkeeping-repair.json` preserve the limitations, including absent pre-repair runner-hash recording.

Review: root non-author OpenAI-family prospective review/release accepted this exact frozen single case (`root-prospective-review.json`, `root-release.json`). Final root scientific acceptance remains separate; the harness author is not its sole reviewer. Original thin sampling and native/full-suite NO_GO outcomes remain unchanged. No different-family/human, full workflow, prism refinement/convergence or release acceptance is claimed.

## Source-reference versus approximate-boundary physics — Read finding

The unchanged physical helper applies `stress @ source_normal` as a constant traction on each **numerical mesh** boundary group and compares its force/moment against the **source** face area/centroid. Its energy denominator is `source.volume * (strain:stress)`. Installed ANYsolid integrates traction using mesh `FacetBasis`; pressure uses numerical facet normals `w.n`; stiffness integrates the numerical tetrahedra. Thus the source-reference affine checks are a combined geometry-and-physics gate. Passing a nonzero geometric boundary tolerance alone cannot guarantee their 1e-9 residual / 1e-10 work tolerances.

For constant affine strain on the numerical domain, exact discrete energy is `V_mesh * (strain:stress)`, so relative source-reference energy error is `abs(V_mesh/V_source - 1)` even with exact assembly. Exact affine equilibrium requires `stress @ n_mesh` on each numerical facet. If `n_source != n_mesh`, the prescribed source-normal traction is a different load; geometry errors can also change areas and moments. The cap-normal tension exact solution similarly assumes the corresponding ideal prism boundary. This pentagon's approximately machine-exact boundary allows both strict gates to pass.

Read-only cross-check of `evidence/thin-resolution-v2-20261028-a1/cases/thin_sampling_aligned/result.json` found volume error 8.085925484735412e-9 and P1/P2 energy errors 8.085925706780017e-9 / 8.085923375311665e-9. That agreement supports a geometry-reference mismatch explanation; it does not waive the observed failures or prove the absence of all other defects. The reported affine residuals 2.9829611212508173e-8 / 2.6701873079574175e-8 and face-resultant error about 5.65682846e-8 likewise exceed the original limits. Sampling/conditioning passing does not override them.

A possible **future prospective correction** is to retain source geometry acceptance at its physical tolerance, independently test algebraic FE identities against the actual numerical-domain normals/areas/volume, and retain a separately named source-reference geometry/physics discrepancy criterion. That changes the reference question and requires root scientific adoption and new registration; no helper, criterion, prior failure or execution authority was changed here. The current stricter conjunction is coherent if intended to demand much closer boundary agreement than the nominal admission tolerance, but it must not be described as an isolated algebraic solver test.
