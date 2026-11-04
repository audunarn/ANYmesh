# Project/mesher integration coverage

The eleven tests originally in ANYmesh `test_authored_child_project_cells.py`
and `test_authored_project_references.py` now live unchanged in ANYfem as
`test_mesher_authored_child_project_cells.py` and
`test_mesher_authored_project_references.py`. They require the real Project,
reference-scope owner and caller dependencies, rather than a reverse runtime
dependency from the mesher library to the application.

The caller workflow runs them in its existing Windows/Linux full suites and
four dedicated macOS integration lanes, retaining all twelve nominal OS/Python
combinations. A mandatory JUnit gate requires all eleven original cases to pass;
missing, duplicate, failed, errored or skipped cases fail that gate. The existing
Windows/Linux full-suite, licensing, parity and qualified-planner gates remain.
The new macOS lanes establish only these integration cases, not full ANYfem
qualification. Current coordinated delivery requires the matching geometry,
mesher and caller CI results before advancing main.

These are reference/ownership adapter tests. They preserve their original
limitations, including `publication_qualified=False`; they do not establish
full meshing, solver or performance qualification. Numerical source, frozen
benchmark evidence, tolerances and budgets are unchanged.
