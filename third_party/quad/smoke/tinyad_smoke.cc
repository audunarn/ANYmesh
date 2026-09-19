// Minimal, self-contained TinyAD smoke test (pinned TinyAD 2.x API).
//
// Proves the vendored TinyAD + Eigen sources compile and compute a correct
// first derivative of a small scalar objective.  Deliberately tiny and
// CPU-light so it can be built and run with bare cl.exe + Windows SDK.
//
// Objective over 2 scalar variables:  f = 1.0*x0^2 + 3.0*x1^2
// At x = (1, -2):  f = 13 and gradient = (2, -12).

#include <cmath>
#include <cstdio>
#include <Eigen/Dense>
#include <TinyAD/ScalarFunction.hh>

int main()
{
    using T = double;
    const auto handles = TinyAD::range(2);  // variable handles: 0, 1

    auto func = TinyAD::scalar_function<1, T>(handles);

    // One element (valence 2) referencing both variables.
    // Return type is inferred as the active TinyAD scalar type, so the
    // per-element expression is differentiated automatically.
    func.add_elements<2>(TinyAD::range(1), [&](auto& element) {
        auto v0 = element.variable(0);
        auto v1 = element.variable(1);
        return v0 * v0 + 3.0 * v1 * v1;
    });

    // variable_dimension == 1 -> readback returns a 1x1 Eigen vector.
    auto one  = Eigen::Matrix<double, 1, 1>(1.0);
    auto neg2 = Eigen::Matrix<double, 1, 1>(-2.0);
    Eigen::VectorXd x = func.x_from_data([&](Eigen::Index h) {
        return (h == 0) ? one : neg2;
    });

    T f;
    Eigen::VectorXd g;
    func.eval_with_gradient(x, f, g);

    const T expected_f = 13.0;
    const T expected_g[2] = {2.0, -12.0};
    const bool ok = std::abs(f - expected_f) < 1e-12 &&
                    std::abs(g[0] - expected_g[0]) < 1e-12 &&
                    std::abs(g[1] - expected_g[1]) < 1e-12;
    std::printf("tinyad_smoke: f=%.6f g=(%.6f, %.6f) | expected f=%.1f g=(%.1f, %.1f) -> %s\n",
                f, g[0], g[1], expected_f, expected_g[0], expected_g[1],
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
