# Test feedback and acceptance

The routine development check is deliberately small:

```powershell
python tools/run_dev_smoke.py
```

It runs actual public meshing, recovery, owner-interface, attachment and atomic
publication controls. On the 2026-10-01 Windows development checkout it ran 15
tests in 2.06 seconds of pytest time (3.78 seconds including startup). This is
feedback for an edit, not a claim that other routes or platforms are accepted.
Run the tests for every changed subsystem alongside the smoke check, for example:

```powershell
python -m pytest -q tests/quad_first_planar
python -m pytest -q tests/quad_first_curved
python -m pytest -q tests/test_native_v2_foundation.py
```

For routine Python edits, the change-aware helper combines the smoke with test
files found through source imports. It shows its selection before running:

```powershell
python tools/run_affected_tests.py --list --paths src/anymesher/recovery.py
python tools/run_affected_tests.py --paths src/anymesher/recovery.py
python tools/run_affected_tests.py --base HEAD~1 --extra tests/test_intersection_meshing.py
```

Without `--paths`, it compares tracked files in the working tree with `--base`
(default `HEAD`) and includes untracked source, test and tool files. Static imports do
not capture every dynamic or external consumer; use `--extra` for relevant tests
the helper did not find. It refuses changed source paths with no test mapping.
It does not schedule the frozen `tests/sg1` gate. A topology or ownership change
may need both planar and curved integration tests; a native change needs native
boundary and runtime checks. This helper is development feedback, not an
acceptance or release command.

The unfiltered source suite remains:

```powershell
python -m pytest
```

GitHub runs the development smoke on each push and pull request. The existing
Windows/Linux/macOS Python 3.11–3.14 source matrix, optional Gmsh matrix,
native wheel smokes and installed Linux wheel suite remain in `Tests`. They run
on a weekly schedule, a manual workflow dispatch, and release-version tags.
To run the full workflow on a candidate branch before acceptance or release:

```powershell
gh workflow run ci.yml --ref <candidate-branch>
```

The full workflow reports its 25 slowest tests per source/installed suite to
support later targeted improvements. Scientific gates, release contracts and
their exact evidence requirements still determine acceptance. A skipped full
job on a routine push is not a passed full job; a scheduled result applies only
to the source and dependency identities it actually tested.
