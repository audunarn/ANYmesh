"""Run advisory, change-aware development checks without consuming a full gate.

Static imports identify likely affected test modules. Dynamic dispatch and optional
workers remain outside this inference; use ``--extra`` for known consumers and run
the repository's full or scientific gates whenever their contracts require them.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from pathlib import Path
import subprocess
import sys

if __package__:
    from .run_dev_smoke import SMOKE_CASES
else:
    from run_dev_smoke import SMOKE_CASES


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "anymesher"
TESTS = ROOT / "tests"


def _relative(path: str | Path) -> str:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"path is outside ANYmesh: {path}")
    return resolved.relative_to(ROOT).as_posix()


def _module(path: Path) -> str:
    parts = path.relative_to(ROOT / "src").with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _source_index() -> dict[str, Path]:
    return {_module(path): path for path in SOURCE.rglob("*.py")}


def _from_base(node: ast.ImportFrom, current: str, package: bool) -> str:
    if node.level == 0:
        return node.module or ""
    parts = current.split(".") if package else current.split(".")[:-1]
    keep = len(parts) - node.level + 1
    if keep < 0:
        return ""
    return ".".join((*parts[:keep], *((node.module or "").split(".") if node.module else ())))


def _exports(index: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Map package-level imported names to the modules defining them."""
    exports: dict[str, dict[str, str]] = {}
    for module, path in index.items():
        if path.name != "__init__.py":
            continue
        names: dict[str, str] = {}
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = _from_base(node, module, package=True)
            for alias in node.names:
                if alias.name == "*":
                    continue
                submodule = f"{base}.{alias.name}"
                names[alias.asname or alias.name] = submodule if submodule in index else base
        exports[module] = names
    return exports


def _imports(
    path: Path, current: str, index: dict[str, Path], exports: dict[str, dict[str, str]]
) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies: set[str] = set()
    package_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in index:
                    dependencies.add(alias.name)
                    if alias.name == "anymesher":
                        package_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _from_base(node, current, package=path.name == "__init__.py")
            if not base.startswith("anymesher"):
                continue
            if base not in index:
                continue
            for alias in node.names:
                if alias.name == "*":
                    dependencies.add(base)
                    continue
                submodule = f"{base}.{alias.name}"
                dependencies.add(
                    submodule if submodule in index else exports.get(base, {}).get(alias.name, base)
                )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in package_aliases:
                target = f"anymesher.{node.attr}"
                dependencies.add(
                    target if target in index else exports.get("anymesher", {}).get(node.attr, "anymesher")
                )
    return dependencies


def _test_modules(index: dict[str, Path], exports: dict[str, dict[str, str]]) -> dict[str, set[str]]:
    result = {}
    for path in TESTS.rglob("test_*.py"):
        name = path.relative_to(ROOT).as_posix()
        if name.startswith("tests/sg1/"):
            continue  # Frozen scientific execution is never scheduled by this helper.
        result[name] = _imports(path, "", index, exports)
    return result


def select_tests(changed: list[str], extra: list[str] = ()) -> tuple[list[str], dict[str, list[str]]]:
    """Return pytest selectors and their source reasons; raise on unknown source.

    The smoke runs even when there are no changed files. The result is advisory,
    not proof that every dynamic consumer or qualification gate was exercised.
    """
    index = _source_index()
    exports = _exports(index)
    consumers: dict[str, set[str]] = defaultdict(set)
    for module, path in index.items():
        # Package export facades import most of the tree. Treating them as
        # ordinary reverse dependencies would select almost every test for
        # every edit, even when a test imports an unrelated exported symbol.
        if path.name == "__init__.py":
            continue
        for dependency in _imports(path, module, index, exports):
            consumers[dependency].add(module)

    test_dependencies = _test_modules(index, exports)
    selected: dict[str, list[str]] = defaultdict(list)
    unknown: list[str] = []
    for raw in changed:
        name = _relative(raw)
        if name.startswith("tests/sg1/"):
            raise ValueError("SG1 test edits require their own scientific gate procedure")
        if name.startswith("tests/") and name.endswith(".py"):
            if (ROOT / name).is_file():
                selected[name].append("edited test")
            continue
        if not name.startswith("src/anymesher/"):
            continue
        source_path = ROOT / name
        if source_path.suffix != ".py" or not source_path.is_file():
            unknown.append(name)
            continue
        start = _module(source_path)
        if start not in index:
            unknown.append(name)
            continue
        affected = {start}
        queue = deque([start])
        while queue:
            for consumer in consumers[queue.popleft()]:
                if consumer not in affected:
                    affected.add(consumer)
                    queue.append(consumer)
        # An __init__ edit can alter every package import and export.
        if source_path.name == "__init__.py":
            affected.update(module for module in index if module.startswith(start + "."))
        matched = [test for test, deps in test_dependencies.items() if deps & affected]
        if not matched:
            unknown.append(name)
        for test in matched:
            selected[test].append(name)

    for raw in extra:
        name = _relative(raw)
        if not name.startswith("tests/") or not name.endswith(".py") or not (ROOT / name).is_file():
            raise ValueError(f"--extra needs an existing test file: {raw}")
        if name.startswith("tests/sg1/"):
            raise ValueError("SG1 tests require their own scientific gate procedure")
        selected[name].append("explicit extra")
    if unknown:
        raise ValueError(
            "No safe affected-test mapping for " + ", ".join(sorted(unknown))
            + "; run explicit pytest files or the full suite"
        )

    full_files = set(selected)
    selectors = [case for case in SMOKE_CASES if case.split("::", 1)[0] not in full_files]
    selectors.extend(sorted(full_files))
    return selectors, dict(selected)


def _git_changed(base: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base, "--"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise ValueError(completed.stderr.strip() or f"git diff failed for {base}")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src/anymesher", "tests", "tools"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if untracked.returncode:
        raise ValueError(untracked.stderr.strip() or "git untracked-file lookup failed")
    ordinary_untracked = (
        line for line in untracked.stdout.splitlines() if not line.startswith("tests/sg1/")
    )
    return sorted(set(completed.stdout.splitlines()) | set(ordinary_untracked))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Git base compared with the working tree")
    parser.add_argument("--paths", nargs="*", help="Explicit changed paths instead of Git diff")
    parser.add_argument("--extra", action="append", default=[], help="Additional affected test file")
    parser.add_argument("--list", action="store_true", help="Show selection without running pytest")
    parser.add_argument("--explain", action="store_true", help="Print every selected file and reason")
    args = parser.parse_args(argv)
    try:
        changed = args.paths if args.paths is not None else _git_changed(args.base)
        selectors, reasons = select_tests(changed, args.extra)
    except ValueError as error:
        parser.error(str(error))
    print(f"Affected development check: {len(reasons)} full test file(s), "
          f"{len(selectors)} pytest selector(s), {len(changed)} changed path(s)", flush=True)
    names = sorted(reasons)
    for name in names if args.explain else names[:20]:
        print(f"  {name}: {', '.join(sorted(set(reasons[name])))}", flush=True)
    if not args.explain and len(names) > 20:
        print(f"  ... {len(names) - 20} more; use --explain", flush=True)
    if args.explain:
        for selector in selectors:
            if selector not in reasons:
                print(f"  {selector}: development smoke", flush=True)
    print("Advisory development feedback only; dynamic consumers and full gates remain separate.", flush=True)
    if args.list:
        return 0
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *selectors], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
