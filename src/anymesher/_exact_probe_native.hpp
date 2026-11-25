// Self-check of the error-free transforms behind the exact predicates.
//
// two_sum / two_diff / two_product / split are exact only with separately
// rounded IEEE operations.  Contraction across statements (GCC's default
// -ffp-contract=fast on FMA targets) breaks split; value-unsafe modes
// (-ffast-math, -fassociative-math, MSVC /fp:fast) can rewrite two_sum.
// setup.py builds with -ffp-contract=off (GCC/Clang) and /fp:precise (MSVC);
// the guards below refuse to compile under an inherited unsafe mode, and
// tests plus the release wheel smoke check the probe's results against exact
// rational arithmetic.  Within-statement contraction (Clang's default "on",
// MSVC /fp:contract) is not detected by the probe: it keeps split exact.
#pragma once

#include <cfloat>

// Supported: the default GCC, Clang and MSVC floating-point semantics.
// setup.py resets the value-unsafe options after inherited flags; what it
// cannot reset is refused here, and run-time damage (for example
// flush-to-zero from a fast-math startup object) is caught by
// anymesher._exact_arithmetic_check.
#if defined(__FAST_MATH__) || defined(__ASSOCIATIVE_MATH__)
#error "anymesher._native needs IEEE arithmetic: build without -ffast-math / -fassociative-math"
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "anymesher._native needs IEEE arithmetic: build without -ffinite-math-only"
#endif
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#error "anymesher._native needs binary64 evaluation (FLT_EVAL_METHOD 0), not x87 excess precision"
#endif
#if defined(_M_FP_FAST)
#error "anymesher._native needs IEEE arithmetic: build without /fp:fast"
#endif

#include "_triangulation_native.hpp"

namespace anymesher_native {

inline PyObject* py_exact_arithmetic_probe(PyObject*, PyObject* args) {
    // Orientation of (ax, ay), (bx, by), (cx, cy) when given, otherwise of a
    // fixed construction from a and b.  The count is checked before any
    // conversion, so every wrong count is a TypeError.
    const Py_ssize_t given = PyTuple_GET_SIZE(args);
    if (given != 2 && given != 8) {
        PyErr_SetString(PyExc_TypeError, "exact_arithmetic_probe takes 2 or 8 arguments");
        return nullptr;
    }
    const bool explicit_points = given == 8;
    double a = 0.0, b = 0.0;
    double ax = 0.0, ay = 0.0, bx = 0.0, by = 0.0, cx = 0.0, cy = 0.0;
    if (!PyArg_ParseTuple(args, "dd|dddddd:exact_arithmetic_probe", &a, &b,
                          &ax, &ay, &bx, &by, &cx, &cy)) return nullptr;
    double product = 0.0, product_tail = 0.0, sum = 0.0, sum_tail = 0.0;
    double high = 0.0, low = 0.0;
    two_product(a, b, product, product_tail);
    two_sum(a, b, sum, sum_tail);
    split(a, high, low);
    const int orientation = explicit_points
        ? expansion_sign(exact_orient_expansion(Point{ax, ay}, Point{bx, by}, Point{cx, cy}))
        : expansion_sign(exact_orient_expansion(Point{a, b}, Point{b, a}, Point{a + b, a - b}));
    return Py_BuildValue("(dddddd i)", product, product_tail, sum, sum_tail, high, low, orientation);
}

}  // namespace anymesher_native
