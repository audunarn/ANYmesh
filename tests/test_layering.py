"""The package must not import its own consumers.

ANYsolver and ANYfileio consume meshes; ANYmesher does not consume solvers or
file readers.  If that arrow ever reverses the family's dependency graph becomes
a cycle, and the packages can no longer be released independently.  The rule is
cheap to state and easy to break by accident -- a debug import added while
chasing a discrepancy is enough -- so it is checked rather than documented.

ANYmaterial is forbidden too, for a different reason.  It would not close a
cycle, but a mesh is topology and geometry: it carries which entity a node came
from, not what the element is made of.  A material import here would be the
first step towards a mesh that cannot be generated without knowing the steel
grade.

The third-party check exists because an undeclared import works fine on the
developer's machine, where every sibling package is installed, and fails on a
user's.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = "anymesher"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE

# Distributions this package may import unconditionally.  Keep in step with the
# `dependencies` list in pyproject.toml -- test_packaging.py checks that.
ALLOWED_THIRD_PARTY = frozenset({"anygeometry", "numpy"})

# Importing any of these would either invert the dependency direction or give a
# mesh an opinion about something that is not its business.
FORBIDDEN = frozenset(
    {
        "anysolver",
        "anyfem",
        "anyfileio",
        "anymaterial",
        "anystructure",
        "anystruct",
    }
)

# Modules allowed one extra module-level import, because the feature is optional
# and the module is only reached when the corresponding extra is installed.
# Anything not listed here must be imported inside the function that needs it,
# so that importing the package never requires an optional dependency.
OPTIONAL_IMPORT_EXCEPTIONS: dict[str, frozenset[str]] = {
    "backends/gmsh.py": frozenset({"gmsh"}),
}


def _modules() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).as_posix()


def _top_level_imports(path: Path) -> set[str]:
    """Every distribution name a module imports, ignoring relative imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # A relative import stays inside this package by construction.
            if node.level:
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_source_tree_is_importable_layout() -> None:
    assert SOURCE_ROOT.is_dir(), f"missing source tree at {SOURCE_ROOT}"
    assert (SOURCE_ROOT / "__init__.py").is_file()
    assert _modules(), "no modules found to check"


def test_no_module_imports_a_consumer() -> None:
    offenders = {}
    for path in _modules():
        forbidden = sorted(_top_level_imports(path) & FORBIDDEN)
        if forbidden:
            offenders[_relative(path)] = forbidden
    assert not offenders, (
        "ANYmesher must not import its consumers, or the dependency graph "
        f"becomes a cycle: {offenders}"
    )


def test_third_party_imports_are_declared() -> None:
    offenders = {}
    for path in _modules():
        allowed = ALLOWED_THIRD_PARTY | OPTIONAL_IMPORT_EXCEPTIONS.get(_relative(path), frozenset())
        undeclared = sorted(
            _top_level_imports(path) - sys.stdlib_module_names - {PACKAGE} - allowed
        )
        if undeclared:
            offenders[_relative(path)] = undeclared
    assert not offenders, (
        "these imports are neither standard library nor declared dependencies; "
        f"add them to pyproject.toml and ALLOWED_THIRD_PARTY, or import them lazily: {offenders}"
    )
