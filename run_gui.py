#!/usr/bin/env python
"""Run the ANYmesher mesher straight from a checkout.

Point an IDE's Run button at this file, or::

    python run_gui.py

``src`` is put on ``sys.path`` first, so this works in a fresh clone with nothing
installed -- which is the whole point of the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    # Prepended so a checkout is what runs, not a stale installed copy: running
    # this file is a statement about *this* working tree.
    sys.path.insert(0, str(_SRC))

from anymesher.gui import main  # noqa: E402  - import follows the path setup

if __name__ == "__main__":
    raise SystemExit(main())
