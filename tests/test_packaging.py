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
    assert geometry_requirements == ["ANYgeometry>=0.2.1,<0.3"]
    assert project["optional-dependencies"]["planar"] == [
        "ANYgeometry[planar]>=0.2.1,<0.3"
    ]


def test_release_workflows_pin_geometry_and_disabled_native_cell() -> None:
    geometry_ref = "37234b7bc6b6c3f2e02cf1c53acb875245d9c3aa"
    ci = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    publish = (REPOSITORY_ROOT / ".github/workflows/publish.yml").read_text(
        encoding="utf-8"
    )

    assert ci.count("repository: audunarn/ANYgeometry") == 4
    assert ci.count(f"ref: {geometry_ref}") == 4
    assert publish.count("repository: audunarn/ANYgeometry") == 1
    assert publish.count(f"ref: {geometry_ref}") == 1
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
    assert ci.count("name: Install Ubuntu Gmsh runtime") == 1
    assert ci.count("if: runner.os == 'Linux'") == 1
    assert ci.count("sudo apt-get update") == 1
    assert ci.count(
        "sudo apt-get install --yes --no-install-recommends libglu1-mesa"
    ) == 1
    assert ci.startswith("name: Tests\n\non:\n  push:\n  pull_request:\n")

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
