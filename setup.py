"""Build the optional lightweight C++17 acceleration module."""

from __future__ import annotations

import os
import sys

from setuptools import Extension, setup


compile_args = ["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3"]
link_args = ["/MANIFEST:NO"] if sys.platform == "win32" else []

extensions = []
if os.environ.get("ANYMESHER_DISABLE_NATIVE", "0") not in {"1", "true", "TRUE"}:
    extensions.append(
        Extension(
            "anymesher._native",
            ["src/anymesher/_native.cpp"],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=True,
        )
    )

setup(ext_modules=extensions)
