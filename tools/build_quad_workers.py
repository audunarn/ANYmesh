"""Build the optional quad-first worker executables from the vendored sources.

The Q4 min-cost-flow worker (LEMON) and the Q5 TinyAD optimizer are separate
processes, not part of the ``anymesher._native`` extension, and are not shipped
in wheels.  This script compiles both from ``third_party/quad`` without network
access, on any platform:

* Windows: ``cl`` from ``PATH`` (run inside a Visual Studio developer shell);
* elsewhere: ``$CXX``, else the first of ``c++``, ``g++`` and ``clang++``.

Outputs follow the single naming rule the Python adapters resolve:
``third_party/quad/worker/out/lemon/quad_mcf_worker[.exe]`` and
``third_party/quad/worker/out/tinyad/quad_tinyad_optimizer[.exe]``.
``-DEIGEN_MPL2_ONLY`` is always set (see ``third_party/quad/ATTRIBUTION.md``).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUAD = REPO / "third_party" / "quad"
VENDOR = QUAD / "vendor"
WORKER = QUAD / "worker"
OUT = WORKER / "out"
SUFFIX = ".exe" if os.name == "nt" else ""

TARGETS = {
    "mcf": (
        WORKER / "quad_mcf_worker.cc",
        OUT / "lemon" / f"quad_mcf_worker{SUFFIX}",
        (VENDOR,),
    ),
    "tinyad": (
        WORKER / "quad_tinyad_optimizer.cc",
        OUT / "tinyad" / f"quad_tinyad_optimizer{SUFFIX}",
        (VENDOR / "tinyad" / "include", VENDOR / "eigen"),
    ),
}


def _compiler() -> list[str]:
    if os.name == "nt":
        if shutil.which("cl") is None:
            raise SystemExit(
                "cl.exe is not on PATH; run from a Visual Studio developer shell"
            )
        return ["cl"]
    for candidate in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if candidate and shutil.which(candidate.split()[0]):
            return candidate.split()
    raise SystemExit("no C++17 compiler found; set CXX")


def _command(compiler: list[str], source: Path, output: Path,
             includes: tuple[Path, ...]) -> list[str]:
    if compiler[0].lower().endswith("cl"):
        return [
            *compiler, "/nologo", "/std:c++17", "/O2", "/EHsc", "/MD", "/W3",
            "/DEIGEN_MPL2_ONLY", *(f"/I{path}" for path in includes),
            str(source), f"/Fo:{output.with_suffix('.obj')}", f"/Fe:{output}",
        ]
    return [
        *compiler, "-std=c++17", "-O2", "-DEIGEN_MPL2_ONLY",
        *(f"-I{path}" for path in includes), str(source), "-o", str(output),
    ]


def build(names: list[str]) -> list[Path]:
    compiler = _compiler()
    built = []
    for name in names:
        source, output, includes = TARGETS[name]
        output.parent.mkdir(parents=True, exist_ok=True)
        command = _command(compiler, source, output, includes)
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=output.parent)
        if not output.is_file():
            raise SystemExit(f"compiler reported success but {output} is missing")
        built.append(output)
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*",
                        help=f"workers to build: {', '.join(TARGETS)} (default: all)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.targets) - set(TARGETS))
    if unknown:
        parser.error(f"unknown worker(s): {', '.join(unknown)}")
    for path in build(list(args.targets) or list(TARGETS)):
        print(f"built {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
