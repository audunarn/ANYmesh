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


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = Path(__file__).resolve().parent

os.chdir(_REPOSITORY_ROOT)
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
