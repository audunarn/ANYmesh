"""Run focused tests against a separate extension without replacing local DLLs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    arguments, pytest_arguments = parser.parse_known_args()
    path = arguments.extension.resolve(strict=True)
    if "anymesher" in sys.modules or "anymesher._native" in sys.modules:
        raise RuntimeError("native qualification requires a fresh process")
    spec = importlib.util.spec_from_file_location("anymesher._native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the separately built native extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    print(f"Native test extension: {path}", flush=True)
    import pytest

    return int(pytest.main(pytest_arguments))


if __name__ == "__main__":
    raise SystemExit(main())
