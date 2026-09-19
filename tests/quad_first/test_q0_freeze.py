"""Focused Q0 freeze tests for the quad-first programme.

No heavy fixtures, no slow diagnostics.  These verify the Q0 invariants that
do not depend on the later milestone runtime: the vendored third-party freeze
(manifest + attribution), and the strict ``QuadMeshingOptions`` contract that
distinguishes the quad-first path from the untouched legacy path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import anymesher
from anymesher.errors import MeshError
from anymesher.quad import QUAD_MESHING_OPTIONS_SCHEMA, QuadMeshingOptions

_REPO = Path(__file__).resolve().parents[2]
_QUAD = _REPO / "third_party" / "quad"
_MANIFEST = _QUAD / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- third-party freeze ----------------------------------------------------

@pytest.fixture(scope="module")
def manifest() -> dict:
    assert _MANIFEST.is_file(), "Q0 third_party/quad/manifest.json is missing"
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def test_manifest_schema_and_policy(manifest: dict) -> None:
    assert manifest["schema"] == "anymesher.quad-third-party-manifest/1"
    scope = manifest["scope"]
    assert "Pinned, vendored" in scope["policy"]
    assert "No install-time fetching" in scope["policy"]


def test_manifest_build_flags(manifest: dict) -> None:
    build = manifest["build"]
    assert build["cpp_standard"] == "C++17"
    assert build["native_abi_unchanged"] is True
    assert "-DEIGEN_MPL2_ONLY" in build["mandatory_compile_flags"]
    evidence = build["toolchain_evidence"]
    assert evidence["smoke_pass"] is True
    assert "14.50" in evidence["compiler"]


def test_components_are_frozen_and_vendored(manifest: dict) -> None:
    components = {c["id"]: c for c in manifest["components"]}
    expected_ids = {"eigen", "tinyad", "libsatsuma", "libtimekeeper", "q-morph", "lemon"}
    assert set(components) == expected_ids
    for comp in manifest["components"]:
        assert comp["pinned_commit"]
        assert comp["license"]
        assert comp["vendored_path"].startswith("vendor/")
        assert int(comp["file_count"]) > 0
        # per-tree SHA-256 is a 64-char lowercase hex digest
        assert len(comp["tree_sha256"]) == 64
        assert comp["tree_sha256"] == comp["tree_sha256"].lower()
        # archive pin must carry a 64-char hex sha256
        assert len(comp["archive"]["sha256"]) == 64


def test_eigen_is_mpl2_and_flagged(manifest: dict) -> None:
    eigen = next(c for c in manifest["components"] if c["id"] == "eigen")
    assert "MPL-2.0" in eigen["license"]
    assert "EIGEN_MPL2_ONLY" in eigen["license"]


def test_exclusions_are_recorded(manifest: dict) -> None:
    excluded = {item["id"] for item in manifest["exclusions"]["items"]}
    assert {"quadwild", "gurobi", "blossom-v", "java"} <= excluded


def test_vendor_attribution_files_present() -> None:
    assert (_QUAD / "ATTRIBUTION.md").is_file()
    assert (_QUAD / "licenses" / "LICENSES.md").is_file()
    attribution = (_QUAD / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "EIGEN_MPL2_ONLY" in attribution
    assert "MPL-2.0" in attribution
    # MIT/Boost donor directories must ship a verbatim LICENSE file.
    for vendor in ("tinyad", "libsatsuma", "libtimekeeper", "q-morph", "lemon"):
        assert (_QUAD / "vendor" / vendor / "LICENSE").is_file(), f"missing {vendor}/LICENSE"


def test_vendor_file_counts_match_manifest() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    for comp in manifest["components"]:
        vendor = _QUAD / comp["vendored_path"]
        actual = sum(1 for p in vendor.rglob("*") if p.is_file())
        assert actual == comp["file_count"], (
            f"{comp['id']}: manifest says {comp['file_count']}, found {actual}"
        )


# --- QuadMeshingOptions strict contract -----------------------------------

def test_defaults_are_conservative_deterministic() -> None:
    opts = QuadMeshingOptions()
    assert opts.seed_mode == "cdt"
    assert opts.orientation == "cross_4theta"
    assert opts.quality_model == "shape"
    assert opts.line_search == "safeguarded"
    assert opts.max_front_iterations == 100_000
    assert opts.max_local_optimizations == 64
    assert opts.cancellation_interval == 256


def test_schema_tag_is_frozen() -> None:
    assert QUAD_MESHING_OPTIONS_SCHEMA == "anymesher.quad-meshing-options/1"


def test_roundtrip_is_stable() -> None:
    opts = QuadMeshingOptions(orientation="boundary_tangent", quality_model="shape_jacobian")
    assert QuadMeshingOptions.from_dict(opts.to_dict()) == opts
    assert QuadMeshingOptions.from_dict(opts.to_dict()).canonical_digest == opts.canonical_digest


def test_digest_is_sha256_hex() -> None:
    digest = QuadMeshingOptions().canonical_digest
    assert len(digest) == 64
    int(digest, 16)  # valid hex


def test_from_dict_rejects_unknown_or_missing_field() -> None:
    d = QuadMeshingOptions().to_dict()
    with pytest.raises(MeshError, match="unknown or missing"):
        QuadMeshingOptions.from_dict({**d, "extra": 1})
    missing = {k: v for k, v in d.items() if k != "seed_mode"}
    with pytest.raises(MeshError, match="unknown or missing"):
        QuadMeshingOptions.from_dict(missing)


def test_from_dict_rejects_schema_mismatch() -> None:
    d = QuadMeshingOptions().to_dict()
    d["schema"] = "anymesher.quad-meshing-options/2"
    with pytest.raises(MeshError, match="schema mismatch"):
        QuadMeshingOptions.from_dict(d)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("seed_mode", "grid"),
        ("orientation", "axis"),
        ("quality_model", "energy"),
        ("line_search", "aggressive"),
    ],
)
def test_enum_fields_are_closed(field: str, bad: str) -> None:
    with pytest.raises(MeshError):
        QuadMeshingOptions(**{field: bad})


@pytest.mark.parametrize(
    "field,guard",
    [
        ("max_front_iterations", 0),
        ("max_front_iterations", -1),
        ("cancellation_interval", 0),
    ],
)
def test_resource_guards_reject_nonpositive(field: str, guard: int) -> None:
    with pytest.raises(MeshError):
        QuadMeshingOptions(**{field: guard})


@pytest.mark.parametrize("value", ["1", 1.5, True, None])
def test_integer_fields_reject_non_int(value: object) -> None:
    with pytest.raises(MeshError):
        QuadMeshingOptions(max_front_iterations=value)


def test_line_search_off_requires_no_local_optimizations() -> None:
    with pytest.raises(MeshError, match="requires max_local_optimizations=0"):
        QuadMeshingOptions(line_search="off", max_local_optimizations=1)
    assert QuadMeshingOptions(line_search="off", max_local_optimizations=0).max_local_optimizations == 0


def test_options_are_frozen() -> None:
    opts = QuadMeshingOptions()
    with pytest.raises(Exception):
        opts.seed_mode = "grid"  # type: ignore[misc]


def test_coerce_discriminates_paths() -> None:
    assert QuadMeshingOptions.coerce(None) is None
    instance = QuadMeshingOptions()
    assert QuadMeshingOptions.coerce(instance) is instance
    from_dict = QuadMeshingOptions.coerce(QuadMeshingOptions().to_dict())
    assert isinstance(from_dict, QuadMeshingOptions)
    assert from_dict == instance
    with pytest.raises(MeshError):
        QuadMeshingOptions.coerce("not-allowed")  # type: ignore[arg-type]


def test_exported_at_top_level() -> None:
    assert anymesher.QuadMeshingOptions is QuadMeshingOptions
    assert anymesher.QUAD_MESHING_OPTIONS_SCHEMA == QUAD_MESHING_OPTIONS_SCHEMA


def test_quad_options_are_independent_of_native() -> None:
    # The legacy native schema tag must remain distinct from the quad schema tag,
    # and NativeMeshingOptions must remain untouched (no quad field).
    import dataclasses
    from anymesher.native_v2 import NativeMeshingOptions

    native_tag = NativeMeshingOptions().to_dict()["schema"]
    assert native_tag == "anymesher.native-meshing-options/1"
    assert native_tag != QUAD_MESHING_OPTIONS_SCHEMA
    assert "QuadMeshingOptions" not in {f.name for f in dataclasses.fields(NativeMeshingOptions)}
    # Independent canonicalisation: same values should NOT collide with native tags.
    assert NativeMeshingOptions().canonical_digest != QuadMeshingOptions().canonical_digest
