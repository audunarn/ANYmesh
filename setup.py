"""Build the optional lightweight C++17 acceleration module."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from setuptools import Extension, setup


def _ensure_msvc_tools_on_path() -> None:
    """Repair setuptools environments that discover MSVC headers but not tools."""

    if sys.platform != "win32" or shutil.which("cl.exe") is not None:
        return
    target = "x64" if sys.maxsize > 2**32 else "x86"
    candidates: list[Path] = []
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(variable)
        if not root:
            continue
        visual_studio = Path(root) / "Microsoft Visual Studio"
        for host in ("HostX64", "HostX86"):
            candidates.extend(
                visual_studio.glob(
                    f"*/*/VC/Tools/MSVC/*/bin/{host}/{target}/cl.exe"
                )
            )
    if not candidates:
        return

    def version_key(compiler: Path) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in compiler.parents[3].name.split("."))
        except ValueError:
            return (0,)

    compiler = max(candidates, key=lambda item: (version_key(item), str(item)))
    os.environ["PATH"] = str(compiler.parent) + os.pathsep + os.environ.get("PATH", "")


_ensure_msvc_tools_on_path()


# The exact predicates use error-free transforms (Dekker split, two_sum,
# two_product) that need separately rounded IEEE operations: a fused
# multiply-add contraction across statements (GCC default on FMA targets)
# breaks split, and value-unsafe modes can reassociate two_sum.  These options
# follow any inherited CFLAGS and reset them (GCC 13: -ffast-math or
# -funsafe-math-optimizations followed by these leave no unsafe macro set and
# the probe exact).
# MSVC: /fp:precise is the default (no contraction); pinned so an inherited
# /fp:fast (CL environment variable) is overridden.  An explicit /fp:contract
# still contracts, but only within one expression, which keeps split exact.
compile_args = ["/std:c++17", "/O2", "/fp:precise"] if sys.platform == "win32" else [
    "-std=c++17", "-O3", "-fno-fast-math", "-fno-unsafe-math-optimizations",
    "-fno-associative-math", "-fno-reciprocal-math", "-ffp-contract=off",
]
link_args = ["/MANIFEST:NO"] if sys.platform == "win32" else []

extensions = []
require_native = os.environ.get("ANYMESHER_REQUIRE_NATIVE", "0") in {
    "1",
    "true",
    "TRUE",
}
if os.environ.get("ANYMESHER_DISABLE_NATIVE", "0") not in {"1", "true", "TRUE"}:
    extensions.append(
        Extension(
            "anymesher._native",
            ["src/anymesher/_native.cpp"],
            depends=[
                "src/anymesher/_triangulation_native.hpp",
                "src/anymesher/_exact_probe_native.hpp",
                "src/anymesher/_quality_pipeline_native.hpp",
                "src/anymesher/_native_v2.hpp",
                "src/anymesher/_native_t3_storage.hpp",
            ],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=not require_native,
        )
    )

setup(ext_modules=extensions)
