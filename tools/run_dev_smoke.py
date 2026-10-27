"""Run a small, real-mesh development check; never claim full acceptance.

Use this after a local edit, followed by tests for the changed subsystem. The
unfiltered ``python -m pytest`` command remains the complete source suite.
"""

from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Keep this selection small and stable. It exercises public linear/quadratic
# routes, native fallback, automatic recovery, planar/curved owner identities,
# structural attachment, and atomic publication with actual source models.
SMOKE_CASES = (
    "tests/test_prepared_corner_binding.py",
    "tests/test_hybrid.py",
    "tests/test_recovery.py",
    "tests/test_native_cpp_boundary.py::test_python_fallback_orientation_and_incidence",
    "tests/quad_first_planar/test_pq3_public_driver.py::test_p01_h_half_q4_dominant_fraction_and_validated_area",
    "tests/quad_first_curved/test_ch3_cylindrical_public.py::test_adjacent_rotated_patches_share_reversed_quadratic_edge",
    "tests/test_structural_pipeline.py::test_reversed_member_boundary_stations_do_not_create_false_eccentric_couplings",
    "tests/quad_first/test_q6_public_integration.py::test_publish_atomically_orders_and_cancels",
)

# Controlled provider/process contracts only. This profile never invokes the
# real optional mesher and is not a provider qualification or release gate.
VOLUME_SMOKE_CASES = (
    'tests/test_volume_mesh.py',
    'tests/test_volume_process.py',
)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--profile', choices=('shell', 'volume'), default='shell')
    options, pytest_args = parser.parse_known_args()
    print(
        "ANYmesher development smoke: focused feedback only; "
        "run affected tests and the required full gates separately",
        flush=True,
    )
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-q",
         *(VOLUME_SMOKE_CASES if options.profile == 'volume' else SMOKE_CASES), *pytest_args],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
