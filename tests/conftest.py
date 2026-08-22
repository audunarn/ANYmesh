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
