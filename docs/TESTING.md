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

Choose the affected files from the change and its dependencies. A change to
topology or ownership may need both planar and curved integration tests; a
native change needs its native boundary and runtime tests. Do not infer that a
smoke pass covers a feature outside its named cases.

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
