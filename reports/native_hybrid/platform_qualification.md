# Native extension platform qualification

## Reproduced locally

- Platform: Windows, CPython 3.13, MSVC C++17.
- Build path: `python setup.py build_ext --inplace` from a Visual Studio developer environment.
- Focused boundary result: 19 passed. The NumPy two-dimensional cross-product warning was subsequently replaced by an explicit scalar determinant, and the affected core plus curved qualification reran warning-free (13 passed).
- Covered behavior: exact predicates, integer PEP 3118 width and signedness, endian rejection, manifold adjacency, scalar/batch parity, chart derivatives/projection, metric construction, constrained smoothing, and edge flipping.

This is Windows-scoped evidence only.

## Automated artifact matrix

The CI and release workflows build compiled wheels with `pypa/cibuildwheel@v4.1.1` for CPython 3.11-3.14 on:

- Windows 64-bit
- manylinux x86-64
- macOS runner architecture

Each wheel is installed by cibuildwheel in its isolated temporary test environment. The smoke installs the checked-out public ANYgeometry dependency, imports ANYmesher from the wheel, asserts that the compiled extension is available, and generates a small mesh. The regular test matrix runs on Windows, Ubuntu, and macOS for the same Python versions. Gmsh remains a separate optional compatibility job and is not part of the production default.

## Remaining evidence

- CI artifact URLs and exact wheel tags must be recorded after the workflow runs.
- Non-Windows behavioral results remain pending until CI reports them.
- Apple Silicon and additional Linux architectures are not claimed by this matrix.
