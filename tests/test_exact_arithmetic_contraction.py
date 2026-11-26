"""The native exact predicates need separately rounded IEEE operations.

GCC's default cross-statement contraction on an FMA target (for example
``-march=haswell`` or aarch64 Linux) breaks Dekker's ``split``; value-unsafe
modes are refused at compile time.  ``setup.py`` builds the extension with
``-ffp-contract=off`` on non-Windows platforms and ``/fp:precise`` on MSVC.
This test checks the compiled transforms against exact rational arithmetic on
deterministic inputs, including exactly collinear constructions and one-ULP
perturbations.  It does not detect within-statement contraction (Clang's
default, MSVC ``/fp:contract``), which keeps ``split`` exact.
"""
import pytest

from anymesher._exact_arithmetic_check import exact_arithmetic_failures


def test_error_free_transforms_are_exact():
    extension = pytest.importorskip("anymesher._native")
    probe = getattr(extension, "exact_arithmetic_probe", None)
    assert callable(probe), "rebuild the extension: this checkout requires exact_arithmetic_probe"
    failures = exact_arithmetic_failures(probe)
    assert not failures, f"{len(failures)} inexact results (fused multiply-add?), first: {failures[:5]}"
