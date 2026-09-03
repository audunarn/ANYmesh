# Third-party notices

ANYmesher's own source code is licensed under MPL-2.0. The dependencies below
are separate works and retain their upstream licenses. Runtime dependencies are
installed separately; their source or object code is not copied into the
ANYmesher wheel. The optional native module is project-owned C++17 code and does
not vendor a third-party meshing library.

| Dependency | Declared requirement | Upstream | License | Distribution |
| --- | --- | --- | --- | --- |
| NumPy | `numpy>=1.26` | https://numpy.org/ | BSD-3-Clause | Separate runtime dependency; not bundled |
| ANYgeometry | `ANYgeometry[planar]>=0.4,<0.5` | https://github.com/audunarn/ANYgeometry | Must be MPL-2.0 before the ANYmesher 0.4.0 release gate | Separate runtime dependency; not bundled |
| ANYtk3D | `ANYtk3D>=0.2.1,<0.3` (`gui3d` extra) | https://github.com/audunarn/ANYtk3D | Verify the installed release metadata | Optional separate dependency; not bundled |
| Gmsh | `gmsh>=4.11` (`gmsh` extra) | https://gmsh.info/ | GPL-2.0-or-later | Optional compatibility backend; separately installed and never selected automatically |
| build | `build>=1.2` (`dev` extra) | https://pypa-build.readthedocs.io/ | MIT | Development/release tool; not bundled |
| pytest | `pytest>=8` (`dev` extra) | https://pytest.org/ | MIT | Development/test tool; not bundled |
| Twine | `twine>=5` (`dev` extra) | https://twine.readthedocs.io/ | Apache-2.0 | Release tool; not bundled |

Release qualification must verify the exact installed dependency metadata and
record any license drift. Dependencies with GPL, AGPL, LGPL, EPL, CDDL,
source-available, non-commercial, or custom terms require deliberate review.
The optional Gmsh extra is intentionally excluded from the default runtime and
from proprietary integration unless separately reviewed.
