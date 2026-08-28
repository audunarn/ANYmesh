"""Executable authority tests for the native ANYmesher release matrix."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_release_authority.py"
DISTRIBUTION = "ANYmesher"
NORMALIZED = "anymesher"
VERSION = "0.3.2"
TAG = f"v{VERSION}"
TERMINAL = "ACCEPTED_ANYMESHER_0_3_2_RELEASE"
LEDGER = Path("docs/release/anymesher-0.3.2-ledger.json")
SDIST = f"{NORMALIZED}-{VERSION}.tar.gz"
CHECKSUM = "ANYmesher-0.3.2-SHA256SUMS.txt"
FROZEN_REQUIREMENTS = (
    "numpy>=1.26",
    "ANYgeometry[planar]>=0.4.1,<0.5",
    'ANYtk3D>=0.5.3,<0.6; extra == "gui3d"',
    'gmsh>=4.11; extra == "gmsh"',
    'build>=1.2; extra == "dev"',
    'pytest>=8; extra == "dev"',
    'twine>=5; extra == "dev"',
)


def _wheel_names() -> list[str]:
    names = []
    for python_tag in ("cp311", "cp312", "cp313", "cp314"):
        names.extend(
            (
                f"{NORMALIZED}-{VERSION}-{python_tag}-{python_tag}-win_amd64.whl",
                f"{NORMALIZED}-{VERSION}-{python_tag}-{python_tag}-"
                "manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                f"{NORMALIZED}-{VERSION}-{python_tag}-{python_tag}-"
                "macosx_11_0_arm64.whl",
            )
        )
    return sorted(names)


def test_release_matrix_freezes_manylinux2014_filenames() -> None:
    linux_wheels = [name for name in _wheel_names() if "manylinux" in name]
    assert linux_wheels == [
        f"{NORMALIZED}-{VERSION}-{python_tag}-{python_tag}-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        for python_tag in ("cp311", "cp312", "cp313", "cp314")
    ]


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Authority Test",
            "-c",
            "user.email=release-authority@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _metadata(
    distribution: str = DISTRIBUTION,
    requirements: tuple[str, ...] | None = None,
) -> bytes:
    requirement_rows = "".join(
        f"Requires-Dist: {requirement}\n"
        for requirement in (
            FROZEN_REQUIREMENTS if requirements is None else requirements
        )
    )
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {VERSION}\n"
        "Requires-Python: >=3.11\n"
        f"{requirement_rows}\n"
    ).encode("utf-8")


def _record_row(name: str, raw: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
    return name, "sha256=" + digest.decode("ascii"), str(len(raw))


def _write_wheel(
    path: Path,
    payload: bytes = b"accepted native build\n",
    *,
    distribution: str = DISTRIBUTION,
    include_native: bool = True,
    corrupt_record: bool = False,
    requirements: tuple[str, ...] | None = None,
) -> None:
    dist_info = f"{NORMALIZED}-{VERSION}.dist-info"
    files = {
        f"{NORMALIZED}/__init__.py": payload,
        f"{dist_info}/METADATA": _metadata(distribution, requirements),
        f"{NORMALIZED}/_native.so": b"native-binary-placeholder\n",
    }
    if not include_native:
        files.pop(f"{NORMALIZED}/_native.so")
    record_name = f"{dist_info}/RECORD"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for name, raw in sorted(files.items()):
        row = _record_row(name, raw)
        if corrupt_record and name.endswith("/__init__.py"):
            row = (name, "sha256=" + "A" * 43, str(len(raw)))
        writer.writerow(row)
    writer.writerow((record_name, "", ""))
    files[record_name] = buffer.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)


def _write_sdist(path: Path, distribution: str = DISTRIBUTION) -> None:
    metadata = _metadata(distribution)
    info = tarfile.TarInfo(f"{NORMALIZED}-{VERSION}/PKG-INFO")
    info.size = len(metadata)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(metadata))


def _write_checksums(assets: Path, names: list[str]) -> None:
    (assets / CHECKSUM).write_text(
        "".join(
            f"{hashlib.sha256((assets / name).read_bytes()).hexdigest()}  {name}\n"
            for name in sorted(names)
        ),
        encoding="ascii",
        newline="\n",
    )


def _run_verifier(
    root: Path,
    mutation: str = "",
) -> subprocess.CompletedProcess[str]:
    repository = root / "r"
    remote = root / "o.git"
    assets = root / "a"
    repository.mkdir(parents=True)
    remote.mkdir()
    assets.mkdir()
    _git(repository, "init", "--quiet")
    _git(remote, "init", "--bare", "--quiet")
    (repository / "source.txt").write_text("frozen source\n", encoding="utf-8")
    source_paths = ["source.txt"]
    if mutation == "textconv-diff-driver":
        (repository / ".gitattributes").write_text(
            "* diff=release-bypass\n", encoding="utf-8"
        )
        source_paths.append(".gitattributes")
    _git(repository, "add", *source_paths)
    _git(repository, "commit", "--quiet", "-m", "freeze artifact source")
    source_commit = _git(repository, "rev-parse", "HEAD")
    source_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    _git(repository, "branch", "-M", "main")
    _git(repository, "remote", "add", "origin", str(remote))
    _git(repository, "push", "--quiet", "-u", "origin", "main")

    attribute_source = ""
    if mutation == "git-attr-source":
        _git(repository, "checkout", "--quiet", "-b", "attack-attributes")
        (repository / ".gitattributes").write_text(
            "* diff=release-bypass\n", encoding="utf-8"
        )
        _git(repository, "add", ".gitattributes")
        _git(repository, "commit", "--quiet", "-m", "attacker attributes")
        attribute_source = _git(repository, "rev-parse", "HEAD")
        _git(repository, "checkout", "--quiet", "main")

    names = _wheel_names()
    if mutation == "missing-wheel":
        names.pop()
    elif mutation == "matrix-platform":
        names[-1] = names[-1].replace("win_amd64", "musllinux_1_2_x86_64")
        names.sort()
    elif mutation == "manylinux-2-28":
        index = next(
            index for index, name in enumerate(names) if "manylinux" in name
        )
        names[index] = names[index].replace(
            "manylinux_2_17_x86_64.manylinux2014_x86_64",
            "manylinux_2_24_x86_64.manylinux_2_28_x86_64",
        )
        names.sort()
    for index, name in enumerate(names):
        requirements = None
        if index == 0:
            if mutation == "active-marker-requirement":
                requirements = FROZEN_REQUIREMENTS + (
                    "malicious-runtime>=1; python_version >= '3.11'",
                )
            elif mutation == "unknown-extra-marker":
                requirements = FROZEN_REQUIREMENTS + (
                    'malicious-runtime>=1; extra == "unknown"',
                )
            elif mutation == "compound-extra-marker":
                requirements = tuple(
                    (
                        'gmsh>=4.11; extra == "gmsh" and '
                        'python_version >= "3.11"'
                    )
                    if requirement == 'gmsh>=4.11; extra == "gmsh"'
                    else requirement
                    for requirement in FROZEN_REQUIREMENTS
                )
            elif mutation == "wrong-extra-marker":
                requirements = tuple(
                    'gmsh>=4.11; extra == "dev"'
                    if requirement == 'gmsh>=4.11; extra == "gmsh"'
                    else requirement
                    for requirement in FROZEN_REQUIREMENTS
                )
            elif mutation == "duplicate-base-requirement":
                requirements = FROZEN_REQUIREMENTS + ("numpy>=1.26",)
            elif mutation == "duplicate-optional-requirement":
                requirements = FROZEN_REQUIREMENTS + (
                    'gmsh>=4.11; extra == "gmsh"',
                )
            elif mutation == "unmarked-extra-requirement":
                requirements = FROZEN_REQUIREMENTS + ("malicious-runtime>=1",)
        _write_wheel(
            assets / name,
            distribution=(
                "DifferentDistribution"
                if mutation == "wrong-wheel-metadata" and index == 0
                else DISTRIBUTION
            ),
            include_native=not (mutation == "missing-native" and index == 0),
            corrupt_record=mutation == "bad-record" and index == 0,
            requirements=requirements,
        )
    _write_sdist(
        assets / SDIST,
        "DifferentDistribution" if mutation == "wrong-sdist-metadata" else DISTRIBUTION,
    )
    artifact_names = sorted([*names, SDIST])
    artifacts = []
    for name in artifact_names:
        raw = (assets / name).read_bytes()
        artifacts.append(
            {
                "bytes": len(raw),
                "filename": name,
                "sha256": hashlib.sha256(raw).hexdigest().upper(),
            }
        )
    ledger = {
        "artifact_source": {"commit": source_commit, "tree": source_tree},
        "artifacts": artifacts,
        "distribution": DISTRIBUTION,
        "publication_authorized": True,
        "qualification": {
            "accepted_terminal": TERMINAL,
            "evidence_sha256": "A" * 64,
            "independent_review_sha256": "B" * 64,
        },
        "schema": "anyecosystem.release-ledger-v1",
        "tag": TAG,
        "version": VERSION,
    }
    if mutation == "wrong-byte-count":
        ledger["artifacts"][0]["bytes"] += 1
    elif mutation == "wrong-terminal":
        ledger["qualification"]["accepted_terminal"] = "REJECTED_RELEASE"
    elif mutation == "evidence-hash":
        ledger["qualification"]["evidence_sha256"] = "0" * 64
    elif mutation == "review-hash":
        ledger["qualification"]["independent_review_sha256"] = "A" * 64
    elif mutation == "wrong-source":
        ledger["artifact_source"]["tree"] = "0" * 40
    elif mutation == "noncanonical-tag-ref":
        ledger["tag"] = f"{TAG}^{{commit}}"

    target = repository / LEDGER
    target.parent.mkdir(parents=True)
    canonical = json.dumps(ledger, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if mutation == "noncanonical-json":
        target.write_text(json.dumps(ledger), encoding="utf-8")
    elif mutation == "duplicate-json-key":
        target.write_text(
            canonical.replace(
                '  "schema": "anyecosystem.release-ledger-v1",\n',
                '  "schema": "anyecosystem.release-ledger-v1",\n'
                '  "schema": "anyecosystem.release-ledger-v1",\n',
            ),
            encoding="utf-8",
            newline="\n",
        )
    elif mutation == "nonfinite-json":
        target.write_text(
            canonical.replace('"publication_authorized": true', '"publication_authorized": NaN'),
            encoding="utf-8",
            newline="\n",
        )
    else:
        target.write_text(canonical, encoding="utf-8", newline="\n")
    _git(repository, "add", LEDGER.as_posix())
    if mutation == "extra-child-path":
        (repository / "unexpected.txt").write_text("extra\n", encoding="utf-8")
        _git(repository, "add", "unexpected.txt")
    _git(repository, "commit", "--quiet", "-m", "docs: authorize release")
    _git(repository, "tag", TAG)
    if mutation != "unmerged-tag-child":
        _git(repository, "push", "--quiet", "origin", "HEAD:main")

    git_directory = Path(_git(repository, "rev-parse", "--git-dir"))
    if not git_directory.is_absolute():
        git_directory = repository / git_directory
    info = git_directory / "info"
    info.mkdir(exist_ok=True)
    if mutation == "moved-tag-ref":
        _git(repository, "tag", "--force", TAG, source_commit)
    elif mutation == "missing-tag-ref":
        _git(repository, "tag", "--delete", TAG)
    elif mutation == "replacement-ref":
        _git(repository, "replace", source_commit, _git(repository, "rev-parse", "HEAD"))
    elif mutation == "graft-file":
        (info / "grafts").write_text(
            _git(repository, "rev-parse", "HEAD") + "\n", encoding="ascii"
        )
    elif mutation == "info-attributes":
        (info / "attributes").write_text(
            "* diff=release-bypass\n", encoding="utf-8"
        )

    _write_checksums(assets, artifact_names)
    invoked_tag = f"{TAG}^{{commit}}" if mutation == "noncanonical-tag-ref" else TAG
    environment = os.environ.copy()
    marker = root / "marker"
    attacker = root / "attacker.py"
    attacker.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('invoked\\n', encoding='utf-8')\n"
        "raise SystemExit(97)\n",
        encoding="utf-8",
    )
    command = shlex.join((sys.executable, str(attacker)))
    attributes = root / "external.attributes"
    attributes.write_text("* diff=release-bypass\n", encoding="utf-8")
    config = root / "external.gitconfig"
    config.write_text("", encoding="utf-8")
    _git(repository, "config", "--file", str(config), "core.attributesFile", str(attributes))
    _git(repository, "config", "--file", str(config), "diff.external", command)
    _git(
        repository,
        "config",
        "--file",
        str(config),
        "diff.release-bypass.textconv",
        command,
    )
    if mutation == "global-attributes-config":
        environment["GIT_CONFIG_GLOBAL"] = str(config)
    elif mutation == "system-attributes-config":
        environment["GIT_CONFIG_SYSTEM"] = str(config)
    elif mutation == "core-attributes-config":
        _git(repository, "config", "core.attributesFile", str(attributes))
        _git(repository, "config", "diff.release-bypass.textconv", command)
    elif mutation == "environment-external-diff":
        environment["GIT_EXTERNAL_DIFF"] = command
    elif mutation == "local-external-diff":
        _git(repository, "config", "diff.external", command)
    elif mutation == "textconv-diff-driver":
        _git(repository, "config", "diff.release-bypass.textconv", command)
    elif mutation == "git-attr-source":
        environment["GIT_ATTR_SOURCE"] = attribute_source
        _git(repository, "config", "diff.release-bypass.textconv", command)
    if mutation == "paired-replacement":
        _write_wheel(assets / names[0], b"replacement native build\n")
        _write_checksums(assets, artifact_names)
    elif mutation == "checksum":
        (assets / CHECKSUM).write_text(
            "0" * 64 + f"  {artifact_names[0]}\n",
            encoding="ascii",
            newline="\n",
        )
    elif mutation == "extra-asset":
        (assets / "unregistered.txt").write_text("extra\n", encoding="utf-8")
    elif mutation == "tag":
        invoked_tag = "v0.3.1"

    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--repository-root",
            str(repository),
            "--ledger",
            LEDGER.as_posix(),
            "--assets",
            str(assets),
            "--output",
            str(root / "d"),
            "--tag",
            invoked_tag,
            "--protected-ref",
            "refs/remotes/origin/main",
            "--expected-terminal",
            TERMINAL,
            "--distribution",
            DISTRIBUTION,
            "--version",
            VERSION,
            "--checksum-name",
            CHECKSUM,
            "--sdist",
            SDIST,
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_release_authority_accepts_exact_native_matrix(tmp_path: Path) -> None:
    completed = _run_verifier(tmp_path / "g")
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "paired-replacement",
        "checksum",
        "extra-asset",
        "tag",
        "wrong-source",
        "unmerged-tag-child",
        "wrong-terminal",
        "evidence-hash",
        "review-hash",
        "wrong-byte-count",
        "wrong-wheel-metadata",
        "wrong-sdist-metadata",
        "missing-native",
        "bad-record",
        "missing-wheel",
        "matrix-platform",
        "manylinux-2-28",
        "extra-child-path",
        "noncanonical-json",
        "duplicate-json-key",
        "nonfinite-json",
        "moved-tag-ref",
        "missing-tag-ref",
        "noncanonical-tag-ref",
        "replacement-ref",
        "graft-file",
        "info-attributes",
        "active-marker-requirement",
        "unknown-extra-marker",
        "compound-extra-marker",
        "wrong-extra-marker",
        "duplicate-base-requirement",
        "duplicate-optional-requirement",
        "unmarked-extra-requirement",
    ],
)
def test_release_authority_rejects_mutation(
    tmp_path: Path, mutation: str
) -> None:
    completed = _run_verifier(tmp_path / "g", mutation)
    assert completed.returncode != 0, mutation
    expected = {
        "active-marker-requirement": "active, compound, or malformed requirement marker",
        "compound-extra-marker": "active, compound, or malformed requirement marker",
        "duplicate-base-requirement": "wheel contains duplicate requirements",
        "duplicate-optional-requirement": "wheel contains duplicate requirements",
        "graft-file": "Git grafts are forbidden",
        "info-attributes": "Git info attributes are forbidden",
        "manylinux-2-28": "wheel platform is outside the frozen matrix",
        "missing-tag-ref": "release tag ref does not resolve to a commit",
        "moved-tag-ref": "release tag ref does not identify the ledger HEAD",
        "noncanonical-tag-ref": "release tag is not canonical",
        "paired-replacement": "committed authority",
        "replacement-ref": "Git replacement objects are forbidden",
        "unknown-extra-marker": "wheel optional-extra requirements differ",
        "unmarked-extra-requirement": "wheel base requirements differ",
        "wrong-extra-marker": "wheel optional-extra requirements differ",
    }
    if mutation in expected:
        assert expected[mutation] in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "core-attributes-config",
        "environment-external-diff",
        "git-attr-source",
        "global-attributes-config",
        "local-external-diff",
        "system-attributes-config",
        "textconv-diff-driver",
    ],
)
def test_release_authority_neutralizes_external_git_configuration(
    tmp_path: Path, mutation: str
) -> None:
    case = tmp_path / "g"
    completed = _run_verifier(case, mutation)
    assert completed.returncode == 0, completed.stderr
    assert not (case / "marker").exists()


def test_release_verifier_scrubs_inherited_git_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("_mesh_release_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = (
        "GIT_ATTR_SOURCE",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    )
    for key in keys:
        monkeypatch.setenv(key, "attacker-controlled")
    environment = module._git_environment()
    assert not set(keys) & set(environment)
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"]
    assert environment["GIT_ATTR_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
