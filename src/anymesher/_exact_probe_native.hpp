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

#if defined(__FAST_MATH__) || defined(__ASSOCIATIVE_MATH__)
#error "anymesher._native needs IEEE arithmetic: build without -ffast-math / -fassociative-math"
#endif
#if defined(_M_FP_FAST)
#error "anymesher._native needs IEEE arithmetic: build without /fp:fast"
#endif

#include "_triangulation_native.hpp"

namespace anymesher_native {

inline PyObject* py_exact_arithmetic_probe(PyObject*, PyObject* args) {
    double a = 0.0, b = 0.0;
    double ax = 0.0, ay = 0.0, bx = 0.0, by = 0.0, cx = 0.0, cy = 0.0;
    if (!PyArg_ParseTuple(args, "dd|dddddd:exact_arithmetic_probe", &a, &b,
                          &ax, &ay, &bx, &by, &cx, &cy)) return nullptr;
    // Orientation of (ax, ay), (bx, by), (cx, cy) when given, otherwise of a
    // fixed construction from a and b.
    const bool explicit_points = PyTuple_GET_SIZE(args) == 8;
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
