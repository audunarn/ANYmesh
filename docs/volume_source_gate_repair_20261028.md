# Source gate prerequisites —2026-10-28

Fresh Edit scope: recovery implementation+focusedtests, disposable realpinnedANYfem harness, this note. Initialrepository2f9a331 clean; no sibling writes. Historicalsource900s campaign655.584s closed, no rerunauthority. New checkledger1200s, each<=300s, threads1, uniqueattempts via established Windows/Linux supervisors; failures/JUnit never overwritten. Fullsuite prohibited until rootadoptsfreshplan.

Plan: diagnose attemptedconforming Q4/T3 retile; fix actualproductionroute preserving shape/area/conformity, transactionallocator/rollback/cancellation; originaltest plus focusednegative/rollback regressions mustpass. Export ANYfem exactcommitted80d18140bb075e26b239f5243c8da4cd09b4ea2c (dirtyprimaryprotected), provision realheadless dependencies in disposableownedruntime with candidateowner precedence; prove bothpreviouslymissingconsumer modules run and fullcollection works withoutlaunchingfullsuite. Scientific/release acceptance remainsroot.

## Early checkpoint

Actual diagnostic completed6.191347181s:7unprotectedQ4/T3interfaces all exhaust. ResidualT3shape refusals remain legitimate; the topology defect is bounded candidate discovery: conformingretile creates a newmidpoint→acceptedquadcorner T3/Q4front, but dryrun examines only the2splitparentedges. Oneparentchild is T3/T3(notfront); pairing theotherchild withits sourcesibling reconstructs collinearcornerunion. Newlocalfront notsearched. Intendededit: retainparentchildfirstorder and addonlytouchedmidpointfrontedges forconformingretile. Preserveallangle/convexity/protectiontests. Factor existingfrontStep staging into sharedhelper so retile+frontmerge publish once; existingtwo-commitroute could leavepartialmutation iffrontStepfails/cancels. Regressionmustcoveractualadmissiblelocaledgeand rollback/cancellationatfinalpublish.

ANYfem exact80d18140bb075e26b239f5243c8da4cd09b4ea2c exported bygitarchive into ownedexports; fullsourcehashmap/archiveidentityretained, primarydirtysourceunused. DisposableLinuxconsumer-runtimecreated underownedroot (3.633671578s), dependencyclosure next. Newledgerused9.825018759s/1200, previous655.584sbudgetnotreused.

## Verified handoff / source freeze

State: **prerequisites implemented and verified, root independent acceptance pending**. OriginalQ4/T3shape/area/conformity test passes; all333quad_first tests passed. Eleven actualANYfem headlessconsumer tests pass with realpinnedconsumer and canonicalcandidateowner identity. Fullcollection-only succeeds:4078tests,0collectionerrors,3unchangedoptional module skips(gmsh,2missingcompiledT3extension lanes). No fullsuite executed and no priorgatefailure waived.

### Recovery cause and actual production fix

Conformingretile creates both a T3/T3splitparenthalf and newSteiner→acceptedcornerT3/Q4front. The olddryrun could examine only the2parentchildedges, which either aren'tfronts or can only recombine originalsourcechildren into a collinearQ4. Candidate search now retains legacyparentchildfirst order and additionally checks only touchedfrontedges incident to the sameSteiner node. All existingconvexity,15degree residualT3,protection,shape and area predicates remain unchanged.

`front.py::_stage_front_step` factors the existingclassification/publicfront-stepstaging path; ordinary `front_step` still calls that path and commitsonce with the original publicsignature/doccontract. Recovery stagesretile+sameproductionfrontmerge in one transaction. Provisionalconsumed triangles are removed frompendingadditions, and frontdelta composition is normalized againstresidentbase. This prevents the priorpartialcommit if laterfrontadvancefails/cancels. New tests cover newlycreated midpointfront, exactlyonepublishedgeneration, forcedfailure afteractualfrontstaging, actualcancellationatfinalcommit and exactdigest/generation/node/cellallocator rollback. Existingnegative/quality/failedretile cases remain asserted.

Ownedfiles: `src/anymesher/quad/recovery.py`, tightlyconnected `front.py`, `tests/quad_first/test_q2_recovery.py`(3newcases), `tools/workflow2_source_gate.py`, this note. Allothernewdirtyfiles visibleatfinish belong toroot/otherowner and were untouched. InitialANYmesh2f9a331 was clean; no siblingcheckout modified and noGitintegration. Productionhashes in summary match the exact333-test successfulcandidate; no productionedit after that pass. `git diff --check` passes (existingLF/CRLFnoticeonly).

### Real consumer prerequisite and reserved pin decision

Export/install used committed ANYfem80d18140... first, never123dirtyprimaryfiles. Publicresolutionfailed because unrelatedANYpost>=1 cannotmeet<.2; that failure isretained. Realecosystemdependencyowners were then exported atimmutablecommits and installed under owned `consumer-runtime`, with all downloads/buildcache underownedcache. CurrentANYfem80d actuallyfailedheadlessimport: `__init__→geometry.sketching→sketcher.controller` eagerlyrequires owner `solve_sketch_edit` absentfromisolatedcandidate (and missingGUIcompanions). Both originalcontract/collection failures are retained, nothidden.

Root reserveddecision adopted **ed845899e7c358b46376bc3ad873327b0267e174**, headless11testscope ONLY. It retains sameprepared_reference_scope blob`d207ab6988d350ceb8087664f01036bd6689c6c3` and sameRegion implementation as80d, with compatibleeagerSketchConstraint/Definition/Plane/solve_sketch imports. UnavailableGUIedit APIs occur only in unexercisedFaceSketchTask methods; **GUIediting is unqualified**. Boundedalternatives:1ebd1df alsohasrequiredscope butolder; c1c8ccb lacks preparedscope andneedsoldowner versions, rejected. NoGUIimplementation copied intoANYgeometry orANYfem; no consumerstub or skippedrequiredmodule.

FinalinstalledrealANYfem165Pythonfiles byte-match committeded84589 export. `consumer-runtime-identity.json` recordsall23installeddistribution versions,direct-source URLs,RECORDhashes,ownerorigins and canonicalANYfemProject/ANYgeometry.GeometryModel alias. `consumer-runtime.lock.txt` pins exactversions. Isolatedcandidategeometry/src+mesher/src are first onPYTHONPATH; realANYfem is loaded from disposablevenv. Final`uv pip check` passes. ExistingfixedANYsolidwheel+scikit-fem12.0.2 were installed for the W2testconsumer lanes so they aren'tnewlyskipped atcollection. NoFEsolve ornativeTetWildcall was made.

Dependencyexportcommits/hashes and completeinstalledartifacts are in summary/exportidentities. ANYsolverfullrepoexport inadvertentlyincluded2.38GBunrelatedfixtures; itsownedprocess tree was intentionallyterminated,partialexport retained UNUSED, and a committedruntime-only archive(metadata/src/packages/results) replaced it. This was ordinaryfilesystempreparation, not a numericalcheck/budgetrenewal. No otherprocess was stopped.

### Checks, failures, authority and exact evidence

All checkprocesses use uniqueattemptfolders under `C:/ANYworkspaceWork/workflow2-20261027/evidence/source-gate-repair-20261028-a1`, established Linuxsession/subreaper supervision via readonlyANYsolidhelper, each<=300s, threadvars1, sourcehash/effectiveruntime/command/timeout receipts and preservedstdout/stderr/JUnit. NoJUnit overwritten. New strict durablebudget ledger is separate fromclosed655.584s campaign.

Successfulverification:

- `python -m pytest -q tests/quad_first/test_q2_recovery.py tests/quad_first/test_q1_front.py`:45passed,4.535178365s (firstcandidate).
- `python -m pytest -q tests/quad_first --durations=12`:**333passed**,14.786010306s (finalproductioncandidate includingnewtests).
- `consumer-runtime/bin/python -m pytest -q tests/test_authored_child_project_cells.py tests/test_authored_project_references.py`:**11passed**,12.111767824s underroot-adoptedpin.
- `consumer-runtime/bin/python -m pytest --collect-only -q tests`:**4078collected**,13.177462444s;3existingoptional module skips only,0errors. This is collection, **not execution/fullgate**.
- `consumer-runtime/bin/python tools/workflow2_source_gate.py fingerprint`:passed5.936437042s;165installedANYfemfiles match committedsource,23distributionspinned,owneridentity/originsasserted.
- Final`uv pip check --python consumer-runtime/bin/python`:passed0.377527094s. Previous20packagecheckbeforeANYsolidaddition passed0.292319569s; finalclosure23packages.

Retainedfailedchecks: publicANYpost dependencyresolution0.729818514s; real80d consumerimport15.984140601s; fullcollection80d19.963965206s (sameunsupportedGUIowner import). All remainfailedrecords; no fourthsameproblem retry. Adoptedpin is a root-reviewed changedprerequisite, then passed. Diagnostic6.191347181s,venvcreation3.633671578s,sourceclosureinstall86.988787285s,adoptedpininstall28.645708768s,ANYsolidtestconsumerinstall3.232586913s are included inledger.

**New supervisedprocess-wall216.58672869/1200s; remaining983.41327131s**. All15supervisedattemptreceipts verifyzeroownedprocesses; no timeout. Remainingtime grants nofullsuiteauthority. Earlier900s sourcecampaign, qualification/meshingcallbudgets and allfailures stayclosed/unchanged. Root mustadoptfreshfullgateplan beforeexecution. No CI/devmetadata was changed, preserving historicalgeometrypins andexistinggates.

Complete exactcommands,cumulativeaccounting,pass/errorcounts,sourceSHA256,consumercommit/artifactpins andfullartifacthashinventory: `.../source-gate-repair-20261028-a1/summary.json`. Livingprogress: `budget.json`; source/runtime proof: `consumer-runtime-identity.json`; frozenversions: `consumer-runtime.lock.txt`.

Review: root inspected atomic staging and touched-midpoint discovery; independent non-author OpenAI review found no blocking repair defect. Root accepts the finite repair checks. Full repository/scientific/release acceptance remains unmet; the later native full gate is NO_GO (see `volume_native_gate_20261028.md`).
