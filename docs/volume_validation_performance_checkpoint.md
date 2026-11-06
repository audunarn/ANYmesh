# Volume admission performance checkpoint — 2026-10-28

State: initial profiling; no source changes. Separate ownership: only `_volume_validation.py`, new `tests/test_volume_validation_fastpath.py`, and this note. Root's dirty volume.py/tests/qualification tooling and build-owner exact-envelope artifacts remain protected.

Question: retained1546-cell box admission costs80.0087s despite fast generation. Competing explanations: per-pair SAT/edge-cross-axis Python dispatch; rational contact fallback; boundary geometry coverage. Bounded first experiment profiles unchanged `validate_cells` on retained binary-coordinate box arrays (orientation permutation only), no provider or FE execution. Outcome chooses a sufficient separation certificate/robust arithmetic fast path while preserving every existing fallback and validity/tolerance rule.

Plan: (1) profile retained validation with exact input/source identity. (2) derive and implement a proved sufficient separation/contact certificate, uncertain predicates fall back unchanged; independent negative/differential and cancellation tests. (3) measure full common production admission on retained box/thin and run relevant unit/process/consumer-contract tests within scope. Root independent review and scientific acceptance remain separate. Budget: <=120s per process, <=600s cumulative fresh profile/unit/performance process-wall; reserve final regression/handoff. No meshing, FE solves, geometry repair/jitter/drop/tolerance changes, Git integration, workers or unrelated edits.

## Early measured checkpoint / candidate proof

Unchanged retained box `validate_cells` profile completed: **96.625s** process-wall, **95.863s** profiled function time,1546cells. `_check_pair`:71708calls /93.196s; `numpy.cross`:2583034calls /72.333s. Evidence `../evidence/validation-fastpath-profile-before-8e91327ec4` retains raw logs, command, inputSHA0c5c7bd2628a5973b484a15e49b5158b3bf58c026261e7070773c100d135bc55, source/runtime identities and verified cleanup. Remaining fresh600s budget **503.375s**.

Actual candidate now adds: (a) face-only axes conservative strict separation before expensive edge-cross axes; margin uses2*_EPS rather than weakening original threshold; (b) filtered orient3d sufficient contact proof for sharedvertex/edge pairs. A containing face of p puts p in one closed halfspace and all q nonshared vertices in the opposite open halfspace; q's intersection with p's halfspace is therefore exactly conv(common). Shared vertices are checked as actual binary-equal coordinate rows. Own-face interior and other-tet exterior signs require robust determinant bounds; uncertain signs use the unchanged original `_check_pair_fallback` SAT/clipping/rational algorithm. No skip-all-adjacency heuristic. Remaining work: independent exact-oracle/differential/negative/cancellation tests, measure retained full common admission box/thin, root source review and final handoff.

## Final implementation and verification handoff

State: **implemented and verified against assigned development criteria; root independent acceptance pending**. Production source frozen at final regression run2026-10-28 07:38:05UTC. Only owned source `_volume_validation.py` changed (68 addedlines), plus new `tests/test_volume_validation_fastpath.py` and this note. No edits to root-owned volume.py/test_volume_mesh.py/qualification tooling, build-owner artifacts, ANYgeometry or primary repositories. Root's existing dirty work remains preserved. No Git integration.

### Algorithm and proof

`_check_pair` first tests the existing eight face axes, with conservative strict separation at `< -2*_EPS`; the original routine uses `< -_EPS`. With normalized float64 coordinates, the extra margin dominates projection roundoff. Face axes are a subset of SAT axes; proving separation needs no edge axes. Ambiguous near-threshold results still use the original routine.

For vertex/edge neighbours, `_certified_face_contact` requires an actual face containing **all** shared coordinate rows. Its own opposite vertex must be proved strictly inside and every other-tet nonshared vertex strictly outside. Convexity then puts the first tet in one closed halfspace; the second tet meets it only in conv(shared vertices). Because that simplex belongs to both tets, it is exactly their intersection. This is a sufficient certificate, not an adjacency exemption.

`_orientation_sign` expands the determinant on translated binary coordinates. Its bound is64machineeps times the absolute-product permanent plus64minimum-normal as an underflow floor; this conservatively exceeds orient3d forward roundoff. Only signs outside the bound certify; zero is **uncertainty**, never inferred coplanarity. Cheap normalized distances only select candidate faces. Exact sign/membership proof makes the decision.

Root independent source review accepted the halfspace proof/filter and identified a float32 boundary. That finding is repaired: **every non-float64 pair uses the original routine**. Two float32 boundary/differential tests pin dispatch and behavior. No normalization cast, acceptance tolerance, source/cell mutation, native provider, geometry repair, jitter or dropped check was added. `_check_pair_fallback` body is byte-identical to the original `_check_pair` body fromHEAD (apart from its function name); the summary records its SHA-256. The original SAT/edge-cross/clipping/rational thresholds and all cell/source/boundary validation remain intact.

### Exact checks and measured results

All check receipts use `C:/ANYworkspaceWork/workflow2-20261027/runtime/Scripts/python.exe`, explicit cwd`.../ANYmesh`, PYTHONPATH ownedANYmesh/src and read-only ownedANYgeometry/src, thread variables1, and established `tools/workflow2_check.py` Windows Job Object supervision. Every process bounded120s; every receipt verifies cleanup and zero remaining owned processes. Full stdout/stderr/commands/source hashes/effective runtime/exit/timing remain in the unique external evidence folders below.

| Check | Result | Precise process-wall seconds | Evidence folder under `../evidence` |
|---|---|---:|---|
|Original retained-box validate_cells cProfile|1546cells admitted;71708paircalls,2583034cross calls|96.625|`validation-fastpath-profile-before-8e91327ec4`|
|Initial fastpath+volume+process slice|85passed|9.25|`validation-fastpath-unit1-9d06e6a975`|
|Original pair routine full common box admission|passed|85.71799999999348|`validation-fastpath-box-baseline-4bfa1869d8`|
|Final fast routine full common box admission|passed|14.65700000000652|`validation-fastpath-box-after-ee6175a840`|
|Final fast routine full common thin admission|passed|33.905999999988126|`validation-fastpath-thin-after-c1411132cf`|
|Final new+existing volume/process/consumer slice|**90passed,0failed,0skipped**|6.73499999998603|`validation-fastpath-final-regressions-7a17b99aa9`|

Original profile:95.863s profiled `validate_cells`,93.196s in `_check_pair`,72.333s in repeated `np.cross`. Full common admission measurements exclude JSON/NPZ output serialization, while process-wall receipts above include them and startup:

- Retained box,388nodes/1546cells/454facets: **84.77993499999866s original →13.78311659998144s candidate**, a **6.1509988967308615x local single-comparison speedup**. Same input SHA`0c5c7bd2628a5973b484a15e49b5158b3bf58c026261e7070773c100d135bc55`, same source/model/region/signature, same validation report and **byte-identical output NPZ** SHA`a473d80d7e82811d19abc30f31bb8afc8aa3ec5b8b54e825875996051881f717`. The protected volume helper source hash matches between runs. Baseline binds the preserved original pair body in the test-only timing helper; no production bypass exists.
- Retained thin,1954nodes/7641cells/2392facets: **32.969021600001724s candidate common admission**. Input SHA`68dfd54e89a2e778a00cb2f45e3f4ec7cfb49270e244b49000aa35242b30f108`, output SHA`21d393fcad5121d8d7b5bd8e7b4ad752700bf12c8840f08e930e09ea2b252d48`. Original thin admission was **not measured**, so no thin speedup is claimed.

Each measurement writes new evidence-only geometry/mesh/results, with atomic final NPZ/JSON replacement and no overwrite of retained input or qualification evidence. Profiling/performance helpers perform **zero native provider calls and zero FE calls**. Existing controlled consumer unit tests exercise assembly/recovery but no FE solve. Previously failed retained qualification/np.int32 serialization evidence remains untouched; these measurements do not accept that campaign.

Final exact test command:

`python -m pytest -q tests/test_volume_validation_fastpath.py tests/test_volume_mesh.py tests/test_volume_process.py tests/test_volume_consumer.py --junitxml=<final evidence folder>/pytest.xml`

Tests cover independently rational-certified filtered signs including underflow and200near-coplanar perturbations,240random positive-cell pair differential decisions with accepted intersections checked rationally, real certified vertex/edge contacts, arbitrary adjacency with positive overlap, extra coplanar nonconforming contact, forced uncertainty fallback, float32 near-contact boundary dispatch, complete thin-cell facet equivalence, source/mesh validation negatives, subprocess cleanup/timeout/cancellation and installed consumer contracts. Existing predicates, tolerances, refused classes and cancellation checks remain unchanged. `git diff --check -- src/anymesher/_volume_validation.py` passed. No failed checks or timeout in this milestone.

### Accounting, review and limits

**Total new milestone process-wall246.89099999997416s /600s; remaining353.10900000002584s.** No budget renewal or consumption of old geometry/full-unit/retained-qualification/ten-shot budgets. No run was restarted.

Machine-readable commands, chronological cumulative accounting, exact input/source/output identities, runtime settings, test counts, original-function body equality and artifact SHA-256 inventory:

`C:/ANYworkspaceWork/workflow2-20261027/evidence/validation-fastpath-worker-summary.json`

Root review found and prompted the float32 guard; final independent root acceptance remains pending. No scientific/FE/native/provider qualification, generalized benchmark claim, provider capability promotion, full mesh-repository/platform suite, package release or publication was performed. This is a measured retained-array admission optimization with focused regression evidence.
