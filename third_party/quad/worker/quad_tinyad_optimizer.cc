// ANYmesh quad-first Q5 TinyAD local-optimization worker.
//
// Solves the free interior Q4-patch re-optimization problem by bounded
// steepest descent with a safeguarded (Armijo) line search on the
// documented composite energy (size + alignment + shape + Jacobian).
// Uses the pinned + vendored TinyAD + Eigen for analytic first derivatives.
//
// Protocol
// --------
// Request (stdin, single JSON object)  anymesher.quad-tinyad-request/1
//   {
//     "schema"        : "anymesher.quad-tinyad-request/1",
//     "h"             : <float>         // target edge length, strictly positive, finite
//     "axis"          : [ux, uy]        // reference axis (normalized internally)
//     "n_nodes"       : <int>           // total nodes, N >= 1
//     "nodes"         : [[x, y], ...]   // N pairs, all finite (flat 2*N array accepted too)
//     "free"          : [i, ...]        // distinct, within [0, N)
//     "quads"         : [[i0, i1, i2, i3], ...]  // four distinct indices in [0, N)
//     "max_iter"      : <int>           // clamped to [1, 1000]
//     "worker_self_test" : "crash" | "hang" | "malformed" | "force_invalid"  // optional
//   }
//
// Response (stdout, single JSON object)  anymesher.quad-tinyad-response/1
//   {
//     "schema"            : "anymesher.quad-tinyad-response/1",
//     "status"            : "CONVERGED" | "NOIMPROVE" | "ERROR",
//     "objective_initial" : <float>,
//     "objective_final"   : <float>,
//     "iterations"        : <int>,
//     "free_nodes"        : [[x, y], ...]   // parallel to request "free"
//     "message"           : <string>        // human-readable, optional
//   }
//
// Energy (see src/anymesher/quad/patch_energy.py::energy for the canonical
// pure-Python reference with the same operation order):
//
//   per quad (p0, p1, p2, p3) with edges e_k and squared norms n_k:
//     size      = (1/4) * sum_{k=0..3} ((n_k / h^2) - 1)^2
//     alignment = (1/4) * sum_k (1 - (e_k . dir_k)^2 / (n_k + eps))
//                 where dir_k = u for k even, v for k odd
//     shape     = (1/4) * sum_{k=0..3} (e_k . e_{k+1 mod 4})^2 / (n_k n_{k+1} + eps)
//     jac       = (1/2) ((e0 x e1) / h^2 - 1)^2 + (1/2) ((e1 x e2) / h^2 - 1)^2
//   total       = sum (size + alignment + shape + jac) over all quads
//
// with u = (ux, uy) (normalized), v = (-uy, ux), eps = 1e-12,
// and (a x b) := a.x b.y - a.y b.x.
//
// Optimizer
// ---------
//   * All nodes are variable handles 0..N-1 (including protected ones); every
//     quad references exactly four distinct corners via add_elements<4>.
//   * After eval_with_gradient the gradient components for protected nodes are
//     zeroed; steepest descent then leaves them fixed.
//   * Direction d = -g (free components only).
//   * Armijo backtracking line search (c1=1e-4, rho=0.5, max 30 contractions).
//   * Per-coordinate clamp to [x0 - 0.5 h, x0 + 0.5 h] before acceptance.
//   * A step is accepted only if the clamped candidate is *valid* (every
//     quad's two triangle determinants strictly > 0); otherwise rollback.
//   * Convergence: accepted step norm < 1e-10 OR free-gradient norm < 1e-10
//     OR max_iter reached. On NOIMPROVE the returned free_nodes are the
//     *initial* values (the caller must re-validate before accepting any
//     solution; see the worker adapter).

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Platform headers stay at file scope: including them inside the anonymous
// namespace below breaks libstdc++ (std::fputs/std::fflush lookups).
#if defined(_WIN32) || defined(WIN32)
// windows.h otherwise defines min/max macros that break Eigen, TinyAD and
// std::max/std::min below.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif

#include <Eigen/Dense>
#include <TinyAD/ScalarFunction.hh>

namespace {

constexpr const char *kRequestSchema  = "anymesher.quad-tinyad-request/1";
constexpr const char *kResponseSchema = "anymesher.quad-tinyad-response/1";

// Energy constants (must match patch_energy.py).
constexpr double kEps   = 1e-12;

// Line-search / optimizer constants.
constexpr double kArmijoC1    = 1e-4;
constexpr double kArmijoRho   = 0.5;
constexpr int    kMaxLineIt   = 30;
constexpr double kStepConv    = 1e-10;
constexpr double kGradConv    = 1e-10;
constexpr int    kDefaultIter = 32;
constexpr int    kMaxIter     = 1000;
constexpr int    kMinIter     = 1;

// ---------------------------------------------------------------------------
// Minimal JSON parser: numbers (int/float with exponent), strings, arrays,
// objects, true/false/null.  Throws std::runtime_error on any parse error.
// ---------------------------------------------------------------------------

class JsonError : public std::runtime_error {
public:
    explicit JsonError(const char *m) : std::runtime_error(m) {}
};

struct JsonValue {
    enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
    bool b = false;
    double d = 0.0;
    std::string s;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue *at(const char *k) const {
        for (const auto &kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const char *p) : p_(p) {}
    JsonValue parse() {
        JsonValue v = value();
        skipWs();
        if (*p_ != '\0') throw JsonError("trailing data");
        return v;
    }

private:
    const char *p_;

    void skipWs() {
        while (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r' ||
               *p_ == '\v' || *p_ == '\f')
            ++p_;
    }

    double num_() {
        skipWs();
        const char *after = p_;
        if (*after == '-' || *after == '+') ++after;
        while (*after >= '0' && *after <= '9') ++after;
        if (*after == '.') { ++after; while (*after >= '0' && *after <= '9') ++after; }
        if (*after == 'e' || *after == 'E') {
            ++after;
            if (*after == '-' || *after == '+') ++after;
            while (*after >= '0' && *after <= '9') ++after;
        }
        if (after == p_) throw JsonError("bad number");
        const double val = std::strtod(p_, nullptr);
        p_ = after;
        return val;
    }

    std::string str_() {
        skipWs();
        if (*p_ != '"') throw JsonError("bad string");
        ++p_;
        std::string out;
        while (*p_ != '\0') {
            char c = *p_;
            if (c == '"') { ++p_; return out; }
            if (c == '\\') {
                ++p_;
                char e = *p_;
                ++p_;
                switch (e) {
                    case '"' : out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/' : out.push_back('/');  break;
                    case 'n' : out.push_back('\n'); break;
                    case 't' : out.push_back('\t'); break;
                    case 'r' : out.push_back('\r'); break;
                    case 'b' : out.push_back('\b'); break;
                    case 'f' : out.push_back('\f'); break;
                    default:  throw JsonError("bad escape");
                }
            } else {
                out.push_back(c);
                ++p_;
            }
        }
        throw JsonError("unterminated string");
    }

    JsonValue arr_() {
        JsonValue v;
        v.kind = JsonValue::ARR;
        ++p_;
        skipWs();
        if (*p_ == ']') { ++p_; return v; }
        for (;;) {
            v.arr.push_back(value());
            skipWs();
            if (*p_ == ',') { ++p_; skipWs(); continue; }
            if (*p_ == ']') { ++p_; break; }
            throw JsonError("bad array");
        }
        return v;
    }

    JsonValue obj_() {
        JsonValue v;
        v.kind = JsonValue::OBJ;
        ++p_;
        skipWs();
        if (*p_ == '}') { ++p_; return v; }
        for (;;) {
            JsonValue key = value();
            if (key.kind != JsonValue::STR) throw JsonError("bad key");
            skipWs();
            if (*p_ != ':') throw JsonError("expected ':'");
            ++p_;
            v.obj.emplace_back(std::move(key.s), value());
            skipWs();
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == '}') { ++p_; return v; }
            throw JsonError("bad object");
        }
    }

    JsonValue value() {
        skipWs();
        char c = *p_;
        if (c == '{') return obj_();
        if (c == '[') return arr_();
        if (c == '"') { JsonValue v; v.kind = JsonValue::STR; v.s = str_(); return v; }
        if (c == 'f') { if (std::strncmp(p_, "false", 5) != 0) throw JsonError("bad literal");
                        JsonValue v; v.kind = JsonValue::BOOL; v.b = false; p_ += 5; return v; }
        if (c == 't') { if (std::strncmp(p_, "true", 4) != 0) throw JsonError("bad literal");
                        JsonValue v; v.kind = JsonValue::BOOL; v.b = true;  p_ += 4; return v; }
        if (c == 'n') { if (std::strncmp(p_, "null", 4) != 0) throw JsonError("bad literal");
                        JsonValue v; v.kind = JsonValue::NUL; p_ += 4; return v; }
        if (c == '-' || (c >= '0' && c <= '9') || c == '.') {
            JsonValue v; v.kind = JsonValue::NUM; v.d = num_(); return v;
        }
        throw JsonError("unexpected token");
    }
};

// ---------------------------------------------------------------------------
// Typed accessors (throw JsonError with a message).
// ---------------------------------------------------------------------------

struct Request {
    double h = 0.0;
    double ux = 1.0, uy = 0.0;
    long long n_nodes = 0;
    std::vector<double> nx, ny;                 // length n_nodes, finite
    std::vector<long long> free;                // sorted, distinct, in [0, n_nodes)
    std::vector<std::array<long long, 4>> quads; // length Q, 4 distinct in [0, n_nodes)
    int max_iter   = kDefaultIter;
    std::string self_test;
};

const JsonValue *req_field(const JsonValue &o, const char *name, const char *ctx) {
    const JsonValue *f = o.at(name);
    if (!f) throw JsonError(ctx);
    return f;
}

long long to_int(const JsonValue &v, const char *ctx) {
    if (v.kind != JsonValue::NUM) throw JsonError(ctx);
    if (!std::isfinite(v.d)) throw JsonError(ctx);
    const double i = std::floor(v.d);
    if (i != v.d) throw JsonError(ctx);
    if (i < -9e15 || i > 9e15) throw JsonError(ctx);
    return static_cast<long long>(i);
}

double to_double(const JsonValue &v, const char *ctx) {
    if (v.kind != JsonValue::NUM) throw JsonError(ctx);
    if (!std::isfinite(v.d)) throw JsonError(ctx);
    return v.d;
}

std::vector<double> to_double_arr(
        const JsonValue &v, const char *ctx, size_t expected = 0) {
    if (v.kind != JsonValue::ARR) throw JsonError(ctx);
    if (expected && v.arr.size() != expected) throw JsonError(ctx);
    std::vector<double> out;
    out.reserve(v.arr.size());
    for (const JsonValue &x : v.arr)
        out.push_back(to_double(x, ctx));
    return out;
}

std::vector<long long> to_int_arr(const JsonValue &v, const char *ctx,
                                  size_t expected = 0) {
    if (v.kind != JsonValue::ARR) throw JsonError(ctx);
    if (expected && v.arr.size() != expected) throw JsonError(ctx);
    std::vector<long long> out;
    out.reserve(v.arr.size());
    for (const JsonValue &x : v.arr) out.push_back(to_int(x, ctx));
    return out;
}

void clamp_max_iter(int &x) { x = std::max(kMinIter, std::min(kMaxIter, x)); }

// ---------------------------------------------------------------------------
// Response writing.
// ---------------------------------------------------------------------------

std::string json_str(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:
                if (c < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
    return out;
}

std::string format_double(double x) {
    if (!std::isfinite(x))
        throw std::runtime_error("non-finite value in worker response");
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", x);
    return buf;
}

bool write_response(
        const char *status, double f0, double f1, long long iters,
        const std::vector<double> &free_nodes, int *rc,
        const char *message = nullptr) {
    std::string out = "{\"schema\":\"" + std::string(kResponseSchema) + "\"";
    out += ",\"status\":\"" + std::string(status) + "\"";
    out += ",\"objective_initial\":" + format_double(f0);
    out += ",\"objective_final\":" + format_double(f1);
    out += ",\"iterations\":" + std::to_string(iters);
    out += ",\"free_nodes\":[";
    for (std::size_t i = 0; i + 1 < free_nodes.size(); i += 2) {
        if (i) out += ",";
        out += "[" + format_double(free_nodes[i]) + "," + format_double(free_nodes[i + 1]) + "]";
    }
    out += "]";
    if (message)
        out += ",\"message\":" + json_str(message);
    out += "}";
    std::fputs(out.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    *rc = 0;
    return true;
}

int json_fail(const char *msg, const std::string &extra = "") {
    std::string m = std::string("quad_tinyad_optimizer: ") + msg;
    if (!extra.empty()) m += " " + extra;
    int rc = 0;
    write_response("ERROR", 0.0, 0.0, 0, std::vector<double>{}, &rc, m.c_str());
    return rc;
}

// ---------------------------------------------------------------------------
// Geometry helpers (passive doubles).
// ---------------------------------------------------------------------------

inline double det2(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

// Per-quad validity: both triangle determinants strictly > 0.
bool quad_valid_dets(double p0x, double p0y, double p1x, double p1y,
                     double p2x, double p2y, double p3x, double p3y,
                     double &d0, double &d1) {
    d0 = det2(p1x - p0x, p1y - p0y, p2x - p0x, p2y - p0y);
    d1 = det2(p2x - p1x, p2y - p1y, p3x - p1x, p3y - p1y);
    return d0 > 0.0 && d1 > 0.0;
}

// Whole-field validity: every quad must be valid.
bool all_quads_valid(
        const std::vector<double>& px,
        const std::vector<double>& py,
        const std::vector<std::array<long long,4>>& quads)
{
    for (const auto &q : quads) {
        double d0, d1;
        if (!quad_valid_dets(px[q[0]], py[q[0]], px[q[1]], py[q[1]],
                             px[q[2]], py[q[2]], px[q[3]], py[q[3]], d0, d1)) {
            return false;
        }
    }
    return true;
}

inline double clamp_coord(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

// ---------------------------------------------------------------------------
// Self-test hooks (used by Python-side lifecycle tests).
// ---------------------------------------------------------------------------

#if defined(_WIN32) || defined(WIN32)
static void worker_crash() {
    SetErrorMode(SEM_NOGPFAULTERRORBOX);
    volatile int *p = nullptr;
    *p = 0;
}
static void worker_hang()  { Sleep(INFINITE); }
#else
static void worker_crash() {
    std::signal(SIGSEGV, SIG_DFL);
    std::raise(SIGSEGV);
    for (;;) {}
}
static void worker_hang()  { for (;;) pause(); }
#endif

static void worker_malformed(int *rc) {
    std::fputs("this is not valid json at all {{{\n", stdout);
    std::fflush(stdout);
    *rc = 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// TinyAD objective: composite energy over all quads, using the *full* (free +
// protected) node set as variable handles. Protected nodes are constrained
// later by zeroing their gradient components.
// ---------------------------------------------------------------------------

template <class T>
T quad_energy(
        const std::array<const Eigen::Vector2<T> *, 4> &p,
        double h, const std::array<double, 2> &u)
{
    const T e0x = (*p[1]-(*p[0]))[0];
    const T e0y = (*p[1]-(*p[0]))[1];
    const T e1x = (*p[2]-(*p[1]))[0];
    const T e1y = (*p[2]-(*p[1]))[1];
    const T e2x = (*p[3]-(*p[2]))[0];
    const T e2y = (*p[3]-(*p[2]))[1];
    const T e3x = (*p[0]-(*p[3]))[0];
    const T e3y = (*p[0]-(*p[3]))[1];

    const T n0 = e0x * e0x + e0y * e0y;
    const T n1 = e1x * e1x + e1y * e1y;
    const T n2 = e2x * e2x + e2y * e2y;
    const T n3 = e3x * e3x + e3y * e3y;

    const T h2 = h * h;          // passive * passive
    const T inv4 = 0.25, inv2 = 0.5;

    // Size term.
    const T q0 = n0 / h2 - 1.0;
    const T q1 = n1 / h2 - 1.0;
    const T q2 = n2 / h2 - 1.0;
    const T q3 = n3 / h2 - 1.0;
    const T size = inv4 * (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);

    // Alignment term: edges 0,2 along u; edges 1,3 along v = (-uy, ux).
    const T v0x = -u[1], v0y = u[0];
    const T w00 = e0x * u[0] + e0y * u[1];    // e0 . u
    const T a0  = w00 * w00 / (n0 + kEps);
    const T w11 = e1x * v0x + e1y * v0y;      // e1 . v
    const T a1  = w11 * w11 / (n1 + kEps);
    const T w22 = e2x * u[0] + e2y * u[1];    // e2 . u
    const T a2  = w22 * w22 / (n2 + kEps);
    const T w33 = e3x * v0x + e3y * v0y;      // e3 . v
    const T a3  = w33 * w33 / (n3 + kEps);
    const T align = inv4 * ((1.0 - a0) + (1.0 - a1) + (1.0 - a2) + (1.0 - a3));

    // Shape term: consecutive-edge angle deviation.
    const T d01 = e0x * e1x + e0y * e1y;
    const T shp0 = d01 * d01 / (n0 * n1 + kEps);
    const T d12 = e1x * e2x + e1y * e2y;
    const T shp1 = d12 * d12 / (n1 * n2 + kEps);
    const T d23 = e2x * e3x + e2y * e3y;
    const T shp2 = d23 * d23 / (n2 * n3 + kEps);
    const T d30 = e3x * e0x + e3y * e0y;
    const T shp3 = d30 * d30 / (n3 * n0 + kEps);
    const T shape = inv4 * (shp0 + shp1 + shp2 + shp3);

    // Jacobian term: two-triangle orientation.
    const T detA = e0x * e1y - e0y * e1x;   // (p1-p0) x (p2-p0)
    const T detB = e1x * e2y - e1y * e2x;   // (p2-p1) x (p3-p1)
    const T j0 = detA / h2 - 1.0;
    const T j1 = detB / h2 - 1.0;
    const T jacobian = inv2 * (j0 * j0 + j1 * j1);

    return size + align + shape + jacobian;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    int rc = 0;

    // Optional self-test hooks (in addition to the request-level ones).
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--self-test-crash") == 0) worker_crash();
        if (std::strcmp(argv[i], "--self-test-hang")  == 0) worker_hang();
    }

    std::string input;
    {
        char buf[4096];
        std::size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0)
            input.append(buf, n);
    }

    Request req_typed;
    try {
        JsonParser p(input.c_str());
        const JsonValue r = p.parse();
        if (r.kind != JsonValue::OBJ) throw JsonError("request is not an object");

        // Schema is soft-checked by the caller; warn and continue.
        if (const JsonValue *sc = r.at("schema")) {
            if (sc->kind != JsonValue::STR || sc->s != kRequestSchema)
                std::fprintf(stderr, "quad_tinyad_optimizer: unexpected schema\n");
        }

        // Optional self-test hooks, evaluated before normal solve.
        if (const JsonValue *st = r.at("worker_self_test")) {
            if (st->kind == JsonValue::STR) {
                if (st->s == "crash") worker_crash();
                if (st->s == "hang")  worker_hang();
                if (st->s == "malformed") { worker_malformed(&rc); return 0; }
                if (st->s == "force_invalid") req_typed.self_test = "force_invalid";
            }
        }

        // Required request fields.
        req_typed.h = to_double(*req_field(r, "h", "missing \"h\""),
                                "h must be finite number");
        if (!(req_typed.h > 0)) {
            rc = json_fail("h must be > 0"); return 0;
        }

        const std::vector<double> ax =
            to_double_arr(*req_field(r, "axis", "missing \"axis\""),
                          "axis must be a 2-element array", 2);
        req_typed.ux = ax[0];
        req_typed.uy = ax[1];
        const double nrm2 = req_typed.ux * req_typed.ux + req_typed.uy * req_typed.uy;
        if (!std::isfinite(nrm2) || nrm2 <= 0.0) {
            rc = json_fail("axis must be non-zero finite");
            return 0;
        }
        const double nrm = std::sqrt(nrm2);
        req_typed.ux /= nrm;
        req_typed.uy /= nrm;

        req_typed.n_nodes = to_int(*req_field(r, "n_nodes", "missing \"n_nodes\""),
                                   "n_nodes must be integer");
        if (req_typed.n_nodes < 1) {
            rc = json_fail("n_nodes must be >= 1"); return 0;
        }

        const std::vector<double> nodes =
            to_double_arr(*req_field(r, "nodes", "missing \"nodes\""),
                          "nodes must be a 2*N array", 2 * static_cast<size_t>(req_typed.n_nodes));
        req_typed.nx.resize(req_typed.n_nodes);
        req_typed.ny.resize(req_typed.n_nodes);
        for (size_t i = 0; i < static_cast<size_t>(req_typed.n_nodes); ++i) {
            req_typed.nx[i] = nodes[2 * i];
            req_typed.ny[i] = nodes[2 * i + 1];
        }

        req_typed.free = to_int_arr(*req_field(r, "free", "missing \"free\""),
                                    "free must be an array of integers");
        if (req_typed.free.size() > static_cast<size_t>(req_typed.n_nodes)) {
            rc = json_fail("free has duplicates or out-of-range entries");
            return 0;
        }
        std::sort(req_typed.free.begin(), req_typed.free.end());
        for (std::size_t i = 0; i < req_typed.free.size(); ++i) {
            if (req_typed.free[i] < 0 || req_typed.free[i] >= req_typed.n_nodes) {
                rc = json_fail("free index out of range"); return 0;
            }
            if (i > 0 && req_typed.free[i] == req_typed.free[i - 1]) {
                rc = json_fail("free has duplicates"); return 0;
            }
        }

        const JsonValue *qf = req_field(r, "quads", "missing \"quads\"");
        if (qf->kind != JsonValue::ARR) {
            rc = json_fail("\"quads\" must be an array"); return 0;
        }
        req_typed.quads.reserve(qf->arr.size());
        for (const JsonValue &q : qf->arr) {
            const std::vector<long long> idx =
                to_int_arr(q, "quad must be a 4-element array", 4);
            std::array<long long, 4> a = {idx[0], idx[1], idx[2], idx[3]};
            for (int k = 0; k < 4; ++k) {
                if (a[k] < 0 || a[k] >= req_typed.n_nodes) {
                    rc = json_fail("quad index out of range"); return 0;
                }
                for (int j = 0; j < k; ++j)
                    if (a[k] == a[j]) {
                        rc = json_fail("quad has duplicate corners");
                        return 0;
                    }
            }
            req_typed.quads.push_back(a);
        }

        if (const JsonValue *mi = r.at("max_iter")) {
            int mv = static_cast<int>(to_int(*mi, "max_iter must be integer"));
            clamp_max_iter(mv);
            req_typed.max_iter = mv;
        }
    } catch (const std::exception &e) {
        rc = json_fail(e.what());
        return 0;
    }

    // --- Build the TinyAD objective -----------------------------------------
    // Variable handles: 0..N-1.  Per quad: 4 distinct corners (static valence 4).
    const long long N = req_typed.n_nodes;
    const auto handles = TinyAD::range(static_cast<Eigen::Index>(N));

    auto func = TinyAD::scalar_function<2, double>(handles);
    func.add_elements<4>(
        TinyAD::range(static_cast<Eigen::Index>(req_typed.quads.size())),
        [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
            using T = TINYAD_SCALAR_TYPE(element);
            const long long q = static_cast<long long>(element.handle);
            const auto &quad = req_typed.quads[q];
            const Eigen::Vector2<T> corners[4] = {
                element.variables(quad[0]),
                element.variables(quad[1]),
                element.variables(quad[2]),
                element.variables(quad[3]),
            };
            const std::array<const Eigen::Vector2<T> *, 4> pp = {
                &corners[0], &corners[1], &corners[2], &corners[3]
            };
            return quad_energy(pp, req_typed.h, {req_typed.ux, req_typed.uy});
        });

    // Passive initial positions (free + protected, flattened).
    Eigen::VectorXd x0(2 * N);
    for (long long i = 0; i < N; ++i) {
        x0(2 * i)     = req_typed.nx[static_cast<size_t>(i)];
        x0(2 * i + 1) = req_typed.ny[static_cast<size_t>(i)];
    }

    // Index sets.
    std::vector<Eigen::Index> free_idx;
    free_idx.reserve(req_typed.free.size());
    for (long long f : req_typed.free) {
        free_idx.push_back(2 * f);
        free_idx.push_back(2 * f + 1);
    }
    std::vector<char> is_free(static_cast<size_t>(2 * N), 0);
    for (Eigen::Index i : free_idx) is_free[i] = 1;

    // --- Evaluate initial objective -----------------------------------------
    double f_initial = 0.0;
    try {
        f_initial = func.eval(x0);
    } catch (const std::exception &) {
        rc = json_fail("eval(x0) failed"); return 0;
    }
    if (!std::isfinite(f_initial)) {
        rc = json_fail("initial objective is not finite"); return 0;
    }

    // Helpers.
    const auto initial_free_out = [&]() {
        std::vector<double> o;
        o.reserve(req_typed.free.size() * 2);
        for (long long f : req_typed.free) {
            o.push_back(req_typed.nx[static_cast<size_t>(f)]);
            o.push_back(req_typed.ny[static_cast<size_t>(f)]);
        }
        return o;
    };

    // If there are no free nodes or no quads: nothing to optimize.
    if (free_idx.empty() || req_typed.quads.empty()) {
        write_response("CONVERGED", f_initial, f_initial, 0,
                       initial_free_out(), &rc, "no free nodes or no quads");
        return 0;
    }

    // Force_invalid hook: nudge the first free node outside the allowed box so
    // that no line-search step can be accepted and the caller's re-validation
    // rejects the returned (nudged) positions.
    if (req_typed.self_test == "force_invalid") {
        std::vector<double> out;
        out.reserve(req_typed.free.size() * 2);
        const long long f0i = req_typed.free[0];
        for (long long f : req_typed.free) {
            double px = req_typed.nx[static_cast<size_t>(f)], py = req_typed.ny[static_cast<size_t>(f)];
            if (f == f0i) { px += 2.0 * req_typed.h; py -= 2.0 * req_typed.h; }
            out.push_back(px);
            out.push_back(py);
        }
        write_response("NOIMPROVE", f_initial, f_initial, 0, out, &rc,
                       "force_invalid self-test");
        return 0;
    }

    // Initial-field validity.
    {
        std::vector<double> px(N), py(N);
        for (long long i = 0; i < N; ++i) {
            px[i] = x0(2 * i);
            py[i] = x0(2 * i + 1);
        }
        if (!all_quads_valid(px, py, req_typed.quads)) {
            write_response("NOIMPROVE", f_initial, f_initial, 0,
                           initial_free_out(), &rc,
                           "initial patch is not valid (non-positive triangle determinant)");
            return 0;
        }
    }

    // --- Steepest descent with Armijo backtracking --------------------------
    const double half_h   = 0.5 * req_typed.h;
    const double bound_lo = -half_h, bound_hi = +half_h;

    std::vector<double> px(static_cast<size_t>(N)), py(static_cast<size_t>(N));
    for (size_t i = 0; i < px.size(); ++i) {
        px[i] = req_typed.nx[i];
        py[i] = req_typed.ny[i];
    }

    double f_cur  = f_initial;
    long long iters = 0;
    bool accepted_any = false;

    while (iters < req_typed.max_iter) {
        // Evaluate f + gradient at the current point.
        const size_t ndim = 2 * static_cast<size_t>(N);
        Eigen::VectorXd x(ndim);
        for (size_t i = 0; i < px.size(); ++i) {
            x(2 * i)     = px[i];
            x(2 * i + 1) = py[i];
        }
        Eigen::VectorXd g;
        double f;
        try {
            func.eval_with_gradient(x, f, g);
        } catch (const std::exception &) {
            break;
        }
        if (!std::isfinite(f)) break;

        // Zero protected components.
        for (Eigen::Index i = 0; i < g.size(); ++i)
            if (!is_free[i]) g[i] = 0.0;

        // Free-gradient-norm convergence check.
        double gnorm2 = 0.0;
        for (Eigen::Index i : free_idx) gnorm2 += g(i) * g(i);
        const double gnorm = std::sqrt(gnorm2);
        if (gnorm < kGradConv) break;

        // Armijo backtracking line search.
        double alpha = 1.0;
        bool accepted = false;
        double step_norm = 0.0;

        for (int ls = 0; ls < kMaxLineIt && !accepted; ++ls, alpha *= kArmijoRho) {
            std::vector<double> cx(px.size()), cy(px.size());
            double disp2 = 0.0;
            for (size_t i = 0; i < px.size(); ++i) {
                double nx = px[i] - alpha * static_cast<double>(g(2 * i));
                double ny = py[i] - alpha * static_cast<double>(g(2 * i + 1));
                nx = clamp_coord(nx, req_typed.nx[i] + bound_lo, req_typed.nx[i] + bound_hi);
                ny = clamp_coord(ny, req_typed.ny[i] + bound_lo, req_typed.ny[i] + bound_hi);
                cx[i] = nx;
                cy[i] = ny;
                const double dx = nx - px[i], dy = ny - py[i];
                disp2 += dx * dx + dy * dy;
            }
            step_norm = std::sqrt(disp2);

            // Neighbor-halo validity guard: all quads strictly valid.
            if (!all_quads_valid(cx, cy, req_typed.quads)) continue;

            Eigen::VectorXd xc(ndim);
            for (size_t i = 0; i < px.size(); ++i) {
                xc(2 * i)     = cx[i];
                xc(2 * i + 1) = cy[i];
            }
            double fc = 0.0;
            try {
                fc = func.eval(xc);
            } catch (const std::exception &) {
                continue;
            }
            if (!std::isfinite(fc)) continue;

            // Armijo condition (g is zeroed on protected components).
            if (fc <= f + kArmijoC1 * alpha * gnorm2) {
                px = cx;
                py = cy;
                f_cur = fc;
                accepted = true;
                accepted_any = true;
            }
        }

        if (accepted) {
            ++iters;
            if (step_norm < kStepConv) break;
            continue;
        }

        // No accepted step at all: rollback (positions unchanged); stop.
        break;
    }

    // --- Build response -----------------------------------------------------
    std::vector<double> free_out;
    free_out.reserve(req_typed.free.size() * 2);
    for (long long f : req_typed.free) {
        free_out.push_back(px[static_cast<size_t>(f)]);
        free_out.push_back(py[static_cast<size_t>(f)]);
    }

    const char *status = accepted_any ? "CONVERGED" : "NOIMPROVE";
    write_response(
        status, f_initial, f_cur, iters, free_out, &rc,
        accepted_any ? nullptr : "no strict improvement over the initial objective");

    return 0;
}
