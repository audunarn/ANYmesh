"""Shared pytest configuration.

PyCharm may launch pytest with ``tests`` as the process working directory,
while tests that read repository files do so by relative path.  Normalize the
working directory once for the full session so the suite behaves identically
from PyCharm, PowerShell and CI.

``tests`` is also put on ``sys.path`` so test modules can share helpers by
importing each other, which is how the packaging check reuses the layering
allowlist instead of restating it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = Path(__file__).resolve().parent

os.chdir(_REPOSITORY_ROOT)
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
# Benchmark fixtures (``benchmarks.is1``, ``benchmarks.sg1``) are imported as a
# package from the repository root.  The installed-wheel job runs pytest in
# isolated mode without the root on sys.path; append it at the lowest
# priority.  The root holds no ``anymesher`` package, so the installed
# distribution is still the one imported.
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(_REPOSITORY_ROOT))


_RUN_GUI_TESTS = os.environ.get("ANYMESHER_RUN_GUI_TESTS", "").casefold() in {
    "1",
    "true",
    "yes",
}


def pytest_configure(config: pytest.Config) -> None:
    """Use an isolated basetemp and register the explicit desktop marker."""

    if getattr(config.option, "basetemp", None) is None:
        config.option.basetemp = str(
            _REPOSITORY_ROOT / f".pytest_tmp_{uuid4().hex}"
        )
    config.addinivalue_line(
        "markers", "gui: opt-in test that creates a real desktop window"
    )
    config.addinivalue_line(
        "markers",
        "quad_workers: needs the optional quad-first MCF and TinyAD worker "
        "executables (python tools/build_quad_workers.py)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Real Tk tests never run unless the operator explicitly opts in."""

    del config
    if _RUN_GUI_TESTS:
        return
    skipped = pytest.mark.skip(
        reason="real Tk GUI test is opt-in; set ANYMESHER_RUN_GUI_TESTS=1"
    )
    for item in items:
        if item.get_closest_marker("gui") is not None:
            item.add_marker(skipped)


_REQUIRE_QUAD_WORKERS = os.environ.get("ANYMESHER_REQUIRE_QUAD_WORKERS", "") == "1"


def _missing_quad_workers() -> tuple[str, ...]:
    """Names of the optional quad-first worker executables that are absent."""

    from anymesher.quad import quad_mcf_worker, quad_tinyad_worker

    missing = []
    for name, module, error in (
        ("quad_mcf_worker", quad_mcf_worker, quad_mcf_worker.WorkerNotFound),
        ("quad_tinyad_optimizer", quad_tinyad_worker, quad_tinyad_worker.WorkerNotFoundQ5),
    ):
        try:
            module.default_worker_path()
        except error:
            missing.append(name)
    return tuple(missing)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip worker-dependent tests explicitly, or fail when CI requires them."""

    if item.get_closest_marker("quad_workers") is None:
        return
    missing = _missing_quad_workers()
    if not missing:
        return
    message = (
        f"quad-first worker(s) not built: {', '.join(missing)}; "
        "run `python tools/build_quad_workers.py`"
    )
    if _REQUIRE_QUAD_WORKERS:
        pytest.fail(f"{message} (ANYMESHER_REQUIRE_QUAD_WORKERS=1)", pytrace=False)
    pytest.skip(message)
