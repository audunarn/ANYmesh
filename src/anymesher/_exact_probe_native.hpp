// Self-check of the error-free transforms behind the exact predicates.
//
// two_sum / two_product / split are exact only when every operation rounds
// separately.  A fused multiply-add contraction (for example clang's default on
// arm64) turns ``splitter * value - value`` in split into one rounding and the
// low part of two_product is then wrong.  The extension is built with
// -ffp-contract=off on non-Windows platforms; tests call this probe and check
// its results against exact rational arithmetic.
#pragma once

#include "_triangulation_native.hpp"

namespace anymesher_native {

inline PyObject* py_exact_arithmetic_probe(PyObject*, PyObject* args) {
    double a = 0.0, b = 0.0;
    if (!PyArg_ParseTuple(args, "dd:exact_arithmetic_probe", &a, &b)) return nullptr;
    double product = 0.0, product_tail = 0.0, sum = 0.0, sum_tail = 0.0;
    double high = 0.0, low = 0.0;
    two_product(a, b, product, product_tail);
    two_sum(a, b, sum, sum_tail);
    split(a, high, low);
    const int orientation = expansion_sign(exact_orient_expansion(
        Point{a, b}, Point{b, a}, Point{a + b, a - b}));
    return Py_BuildValue("(dddddd i)", product, product_tail, sum, sum_tail, high, low, orientation);
}

}  // namespace anymesher_native
