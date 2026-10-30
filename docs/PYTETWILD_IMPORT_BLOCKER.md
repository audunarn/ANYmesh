# Pinned provider import blocker

Update: the user approved the documented import-only patch. Separately hashed
Windows and Linux compatibility wheels were built; all native bytes were verified
unchanged. Imports now work without PyVista/VTK. Native box execution remains
unresolved after bounded Windows/Linux timeouts; see WORKFLOW_2_HANDOFF.md.

Observed 2026-10-27 in a fresh Windows CPython 3.11 environment after ordinary
`pip install pytetwild==0.4.2`, without PyVista/VTK. The official wheel imports
`pytetwild._accessor` unconditionally and that imports `pyvista`. Import raises
`ModuleNotFoundError: No module named 'pyvista'` before any meshing is executed.
METADATA requires NumPy and declares PyVista only for the optional `all` extra.

Evidence: C:/ANYworkspaceWork/workflow2-20261027/evidence/feasibility-box-c30b1ddd82.
Process wall 0.703 seconds, exit 1, owned process cleanup verified. No tetrahedral
solve or raw mesh exists. This is a packaging failure, not geometry failure.
Installed 0.4.2 and upstream tags 0.4.1/0.4.0/current main all import the accessor.

## Concrete proposed correction (requires user scope decision)

Use a locally identified compatibility wheel based on official 0.4.2. Change
only the optional accessor import in pytetwild/__init__.py:

```python
from importlib.util import find_spec
if find_spec('pyvista') is not None:
    from . import _accessor as _accessor
```

Retain the existing raw-array API imports. No native extension, meshing option,
algorithm, geometry or acceptance criterion changes. Record original wheel hash,
exact patch, patched artifact hash and recomputed wheel RECORD. Identify this as
a locally patched upstream artifact, not official unmodified 0.4.2. Keep the
original failed evidence. Do not publish the wheel or change the main ANYsolid
qualification environment. Qualification must bind the actual patched artifact.

Alternative: keep official 0.4.2 blocked pending an upstream correction. Installing
PyVista/VTK would enlarge the optional route and would not prove the intended
lightweight backend install. At the time of the original proposal no workaround
had yet been implemented or accepted; the update above records its disposition.
