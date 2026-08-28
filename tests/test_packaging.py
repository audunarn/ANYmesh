"""Packaging metadata has to agree with the code it describes.

A version that drifts from pyproject.toml is invisible until a release is cut
with the wrong number on it, and an allowlist that drifts from the declared
dependencies turns the layering check into decoration.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

import anymesher
from anymesher.errors import MeshError
from anymesher.native_cpp import COMPILED_TRIANGULATION_AVAILABLE
from anymesher.triangulation import constrained_planar_triangulation
from test_layering import ALLOWED_THIRD_PARTY, OPTIONAL_IMPORT_EXCEPTIONS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def _pyproject() -> dict:
    return tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _declared_dependencies() -> set[str]:
    project = _pyproject()["project"]
    requirements = list(project.get("dependencies", ()))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    names = set()
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement.strip())
        if match:
            names.add(match.group(0).lower().replace("_", "-"))
    return names


def test_version_matches_pyproject() -> None:
    assert anymesher.__version__ == _pyproject()["project"]["version"]


def test_release_metadata_is_0_3_2_alpha() -> None:
    project = _pyproject()["project"]
    assert project["version"] == "0.3.2"
    assert project["requires-python"] == ">=3.11"
    assert "Development Status :: 3 - Alpha" in project["classifiers"]


def test_distribution_name_differs_from_the_repository_name() -> None:
    # The repository is ANYmesh; `anymesh` was taken on PyPI.  Asserted rather
    # than assumed, because a well-meaning rename back to `ANYmesh` would
    # collide with an unrelated project and only fail at upload time.
    assert _pyproject()["project"]["name"] == "ANYmesher"


def test_allowed_third_party_imports_are_declared_dependencies() -> None:
    declared = _declared_dependencies()
    permitted = set(ALLOWED_THIRD_PARTY)
    for extra in OPTIONAL_IMPORT_EXCEPTIONS.values():
        permitted |= set(extra)
    undeclared = sorted(
        name for name in permitted if name.lower().replace("_", "-") not in declared
    )
    assert not undeclared, (
        "the layering allowlist permits imports that pyproject.toml does not "
        f"install in any extra: {undeclared}"
    )


def test_run_gui_bootstraps_without_an_install() -> None:
    """The IDE Run-button entry point must work in a bare checkout.

    Executed with a run_name other than ``__main__`` so the path bootstrap and
    the import run but the window does not open.
    """

    import runpy

    script = REPOSITORY_ROOT / "run_gui.py"
    assert script.is_file()

    namespace = runpy.run_path(str(script), run_name="not_main")
    assert callable(namespace["main"])
    assert namespace["main"].__module__ == "anymesher.gui"
    assert 'if __name__ == "__main__":\n    raise SystemExit(main())' in script.read_text(
        encoding="utf-8"
    )


def test_anygeometry_release_dependency_floor_is_exact() -> None:
    project = _pyproject()["project"]
    geometry_requirements = [
        requirement
        for requirement in project["dependencies"]
        if requirement.lower().startswith("anygeometry")
    ]
    assert geometry_requirements == ["ANYgeometry[planar]>=0.4.1,<0.5"]
    assert project["optional-dependencies"]["planar"] == []


def test_gui3d_release_dependency_floor_matches_the_candidate_graph() -> None:
    assert _pyproject()["project"]["optional-dependencies"]["gui3d"] == [
        "ANYtk3D>=0.5.3,<0.6"
    ]


def test_release_workflows_pin_geometry_and_disabled_native_cell() -> None:
    ci_geometry_ref = "6a8b023ef6f65805519c96b56e025b4e3b457a1f"
    release_geometry_ref = "6a8b023ef6f65805519c96b56e025b4e3b457a1f"
    ci = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    publish = (REPOSITORY_ROOT / ".github/workflows/publish.yml").read_text(
        encoding="utf-8"
    )
    checkout = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    setup = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
    upload = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    download = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
    cibuildwheel = "pypa/cibuildwheel@4726cd35bb13f7bde50cf2761f2499ac7b3aa32c"
    pypi = (
        "pypa/gh-action-pypi-publish@"
        "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"
    )
    ci_uses = [
        line.split("uses:", 1)[1].strip().split(" #", 1)[0]
        for line in ci.splitlines()
        if "uses:" in line
    ]
    publish_uses = [
        line.split("uses:", 1)[1].strip().split(" #", 1)[0]
        for line in publish.splitlines()
        if "uses:" in line
    ]

    assert ci_uses == [
        checkout,
        checkout,
        setup,
        checkout,
        checkout,
        setup,
        checkout,
        checkout,
        setup,
        checkout,
        checkout,
        cibuildwheel,
        upload,
    ]
    assert publish_uses == [
        checkout,
        setup,
        upload,
        checkout,
        checkout,
        cibuildwheel,
        upload,
        download,
        setup,
        upload,
        upload,
        download,
        pypi,
        checkout,
        setup,
        pypi,
    ]

    assert ci.count("repository: audunarn/ANYgeometry") == 4
    assert ci.count(f"ref: {ci_geometry_ref}") == 4
    assert publish.count("repository: audunarn/ANYgeometry") == 1
    assert publish.count(f"ref: {release_geometry_ref}") == 1
    assert 'ANYMESHER_DISABLE_NATIVE: "1"' in ci
    assert (
        "tests/test_packaging.py::test_disabled_native_build_is_absence_only"
        in ci
    )

    assert ci.count(
        'python -m pip install -e ".[dev,planar]"'
    ) == 1
    assert ci.count(
        'python -m pip install -e ".[dev,gmsh,planar]"'
    ) == 1
    assert ci.count(
        'python -m pip install -e ".[dev]"'
    ) == 1
    assert 'python -m pip install -e ".[dev,gmsh]"' not in ci
    assert ci.count("tools/release_wheel_smoke.py") == 1
    assert ci.count("--expect-version 0.3.2 --require-native") == 1
    assert ci.count("name: Install Ubuntu Gmsh runtime") == 1
    assert ci.count("if: runner.os == 'Linux'") == 1
    assert ci.count("sudo apt-get update") == 1
    assert ci.count(
        "sudo apt-get install --yes --no-install-recommends libglu1-mesa"
    ) == 1
    assert ci.startswith("name: Tests\n\non:\n  push:\n  pull_request:\n")

    assert publish.startswith(
        "name: Build release artifacts\n\non:\n  release:\n"
    )
    assert "types: [published]" in publish
    assert "workflow_dispatch:" in publish
    assert "repository-url: https://test.pypi.org/legacy/" in publish
    assert publish.count("id-token: write") == 2
    assert "pypa/gh-action-pypi-publish@release/v1" not in publish
    assert publish.count(pypi) == 2
    assert "password:" not in publish
    assert "username:" not in publish
    assert "skip-existing:" not in publish
    assert "name: pypi" in publish
    assert "url: https://pypi.org/p/ANYmesher" in publish
    assert "name: testpypi" in publish
    assert "url: https://test.pypi.org/p/ANYmesher" in publish
    assert "name: ANYmesher-0.3.2-pypi-distributions" in publish
    assert "dist/*.whl" in publish
    assert "dist/*.tar.gz" in publish
    assert "permissions:\n  contents: read" in publish
    assert publish.count(
        'os: [windows-latest, ubuntu-latest, macos-latest]'
    ) == 1
    assert publish.count(
        'CIBW_BUILD: "cp311-* cp312-* cp313-* cp314-*"'
    ) == 1
    assert publish.count('CIBW_ENVIRONMENT: "ANYMESHER_REQUIRE_NATIVE=1"') == 1
    assert "expected 12 wheels" in publish
    assert 'expected_pythons = {"cp311", "cp312", "cp313", "cp314"}' in publish
    assert 'name: ANYmesher-0.3.2-release-bundle' in publish
    assert 'sdist = root / "anymesher-0.3.2.tar.gz"' in publish
    assert 'root.glob("anymesher-0.3.2-*.whl")' in publish
    assert '"version": "0.3.2"' in publish
    assert "ANYmesher-0.3.2-SHA256SUMS.txt" in publish
    assert "ANYmesher-0.3.2-release-manifest.json" in publish
    assert "python -m twine check --strict dist/*.whl dist/*.tar.gz" in publish
    assert "expected_base_requirements" in publish
    assert "RECORD self-row must be blank" in publish
    assert "RECORD integrity mismatch" in publish
    assert publish.count("tools/release_wheel_smoke.py") == 1
    assert publish.count("--expect-version 0.3.2 --require-native") == 1
    assert "gh release download \"$RELEASE_TAG\"" in publish
    production = publish.split("\n  publish-production:\n", 1)[1]
    assert (
        "actions/checkout@"
        "3d3c42e5aac5ba805825da76410c181273ba90b1"
    ) in production
    assert (
        "actions/setup-python@"
        "5fda3b95a4ea91299a34e894583c3862153e4b97"
    ) in production
    assert "ref: ${{ github.event.release.tag_name }}" in production
    assert "fetch-depth: 0" in production
    assert "--pattern" not in production
    assert "--protected-ref refs/remotes/origin/main" in production
    assert (
        "--expected-terminal ACCEPTED_ANYMESHER_0_3_2_RELEASE"
        in production
    )
    assert "--sdist anymesher-0.3.2.tar.gz" in production
    assert "tools/verify_release_authority.py" in production
    assert "python -m build" not in production
    assert "packages-dir: dist/" in publish
    assert "github.event.release.prerelease == false" in production
    assert publish.count("if: github.event_name == 'workflow_dispatch'") >= 3
    assert "timeout-minutes:" not in publish


@pytest.mark.skipif(
    os.environ.get("ANYMESHER_DISABLE_NATIVE") != "1",
    reason="requires the fresh disabled-native build cell",
)
def test_disabled_native_build_is_absence_only() -> None:
    assert COMPILED_TRIANGULATION_AVAILABLE is False

    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    result = constrained_planar_triangulation(points, (0, 1, 2, 3))

    assert result.requested_backend == "auto"
    assert result.selected_backend == "python"
    assert result.actual_backend == "python"
    assert result.fallback_reason == "native_capability_absent"
    with pytest.raises(MeshError, match="no native triangulation boundary"):
        constrained_planar_triangulation(
            points,
            (0, 1, 2, 3),
            backend="native",
        )
