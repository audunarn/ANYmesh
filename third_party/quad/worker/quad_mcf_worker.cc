// ANYmesh quad-first Q4 min-cost-flow worker.
//
// Solves a generic, small, pure-integer min-cost-flow instance using the
// pinned + vendored LEMON NetworkSimplex (single source file, no vendored .cc
// files, no JSON library dep). Typed JSON on stdout, diagnostics on stderr.
//
// Request schema  anymesher.quad-mcf-request/1
// Response schema anymesher.quad-mcf-response/1
//
// Request (stdin, single JSON object):
//   {
//     "schema"  : "anymesher.quad-mcf-request/1",
//     "nodes"   : <int>              // number of nodes (N >= 1)
//     "arcs"    : [[u, v, lo, up, c], ...]   // integer quadruple+cost; lo>=0, up>=lo, c>=0
//     "supply"  : [s_0, ..., s_{N-1}]        // positive=source, negative=sink
//   }
//
// Response (stdout, single JSON object):
//   {
//     "schema"    : "anymesher.quad-mcf-response/1",
//     "status"    : "OPTIMAL"|"INFEASIBLE"|"UNBOUNDED"|"ERROR",
//     "total_cost": <int>            // sum c*a*f_a at the returned flow (0 if no flow)
//     "flows"     : [f_0, ..., f_{A-1}]   // parallel to arcs
//   }
//
// Status is a typed contract: OPTIMAL/INFEASIBLE are expected outcomes;
// UNBOUNDED and ERROR are unexpected for bounded integer instances and are
// surfaced so the Python side can raise a typed failure instead of a silent
// fallback.
//
// Self-test hooks (request field "worker_self_test") for lifecycle tests:
//   "crash"  -- writes to a null pointer and dies with a non-zero exit code.
//   "hang"   -- sleeps forever (tests the Python-side timeout + cancellation).
// Any other value is ignored.

#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>

namespace {

int g_rc = 0;

constexpr const char *kRequestSchema       = "anymesher.quad-mcf-request/1";
constexpr const char *kResponseSchema      = "anymesher.quad-mcf-response/1";
constexpr const char *kStatusOptimal       = "OPTIMAL";
constexpr const char *kStatusInfeasible    = "INFEASIBLE";
constexpr const char *kStatusUnbounded     = "UNBOUNDED";
constexpr const char *kStatusError         = "ERROR";

// --------------------------------------------------------------------------
// Minimal JSON parser: integers, strings, arrays, objects, true/false/null.
// Throws std::runtime_error on any parse error; never buffers unbounded.
// --------------------------------------------------------------------------

class JsonError : public std::runtime_error {
public:
    explicit JsonError(const char *msg) : std::runtime_error(msg) {}
};

struct JsonValue {
    enum Kind { NUL, BOOL, INT, STR, ARR, OBJ } kind;
    bool b = false;
    long long i = 0;
    std::string s;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue> > obj;

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
    bool literal(const char *lit, bool val) {
        const size_t n = strlen(lit);
        if (strncmp(p_, lit, n) == 0) { p_ += n; return val; }
        throw JsonError("bad literal");
    }

    long long int_() {
        skipWs();
        bool neg = false;
        if (*p_ == '-') { neg = true; ++p_; }
        else if (*p_ == '+') { ++p_; }
        if (!isdigit(static_cast<unsigned char>(*p_))) throw JsonError("bad int");
        long long v = 0;
        while (isdigit(static_cast<unsigned char>(*p_))) {
            v = v * 10LL + (static_cast<unsigned char>(*p_) - '0');
            ++p_;
        }
        return neg ? -v : v;
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
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'n':  out.push_back('\n'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'u': {
                        if (p_[0] || !isxdigit(static_cast<unsigned char>(p_[0])))
                            throw JsonError("bad \\u");
                        // Parse 4 hex digits (BMP only; we never need surrogates here).
                        unsigned cp = 0;
                        for (int k = 0; k < 4; ++k, ++p_) {
                            char h = *p_;
                            cp = cp * 16u
                               + (isxdigit(static_cast<unsigned char>(h))
                                      ? (isdigit(static_cast<unsigned char>(h))
                                             ? h - '0'
                                             : tolower(static_cast<unsigned char>(h)) - 'a' + 10)
                                      : 0);
                        }
                        // Encode as UTF-8.
                        if (cp < 0x80u) out.push_back(static_cast<char>(cp));
                        else if (cp < 0x800u) {
                            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                        } else {
                            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                        }
                        break;
                    }
                    default: throw JsonError("bad escape");
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
        ++p_;  // consume '['
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
        ++p_;  // consume '{'
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
        if (c == 'f') { JsonValue v; v.kind = JsonValue::BOOL; v.b = false; p_ += 4; return v; }
        if (c == 't') { JsonValue v; v.kind = JsonValue::BOOL; v.b = true;  p_ += 4; return v; }
        if (c == 'n') { JsonValue v; v.kind = JsonValue::NUL; p_ += 4; return v; }
        { JsonValue v; v.kind = JsonValue::INT; v.i = int_(); return v; }
    }
};

// --------------------------------------------------------------------------
// JSON writer: emits only ints, strings, arrays (no nested objects).
// --------------------------------------------------------------------------

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
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
    return out;
}

void write_response(const char *status, long long total,
                    const std::vector<long long> *flows, int *rc,
                    const char *message = nullptr) {
    std::string out = "{\"schema\":\"" + std::string(kResponseSchema) + "\"";
    out += ",\"status\":\"" + std::string(status) + "\"";
    if (flows) {
        out += ",\"total_cost\":";
        char buf[24];
        snprintf(buf, sizeof(buf), "%lld", total);
        out += buf;
        out += ",\"flows\":[";
        for (size_t i = 0; i < flows->size(); ++i) {
            if (i) out += ",";
            snprintf(buf, sizeof(buf), "%lld", (*flows)[i]);
            out += buf;
        }
        out += "]";
    } else if (message) {
        out += ",\"message\":" + json_str(message);
    }
    out += "}";
    std::fputs(out.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    *rc = 0;
}

int json_fail(const char *msg, const std::string &extra = "") {
    std::string m = std::string("quad_mcf_worker: ") + msg;
    if (!extra.empty()) m += " " + extra;
    write_response(kStatusError, 0, nullptr, &g_rc, m.c_str());
    return 0;  // worker reports a typed failure; does not crash
}

}  // namespace

#if defined(_WIN32) || defined(WIN32)
#include <windows.h>
static void worker_crash() {
    // Keep the lifecycle self-test a real process crash without allowing the
    // Windows fault reporter to hold this noninteractive worker open.
    SetErrorMode(SEM_NOGPFAULTERRORBOX);
    volatile int *p = nullptr;
    *p = 0;
}
static void worker_hang()  { Sleep(INFINITE); }
#else
#include <csignal>
#include <unistd.h>
static void worker_crash() {
    std::signal(SIGSEGV, SIG_DFL);
    std::raise(SIGSEGV);
    for (;;) {}
}
static void worker_hang()  {
    for (;;) pause();
}
#endif

int main(int argc, char **argv) {
    // Optional self-test hooks (in addition to the request-level ones).
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--self-test-crash") == 0) worker_crash();
        if (std::strcmp(argv[i], "--self-test-hang")  == 0) worker_hang();
    }

    // Read all of stdin.
    std::string input;
    {
        char buf[4096];
        while (std::size_t n = std::fread(buf, 1, sizeof(buf), stdin))
            input.append(buf, n);
    }

    JsonValue req;
    try {
        JsonParser p(input.c_str());
        req = p.parse();
    } catch (const std::exception &e) {
        return json_fail(e.what());
    } catch (...) {
        return json_fail("unknown parse failure");
    }
    if (req.kind != JsonValue::OBJ) return json_fail("request is not an object");

    // Self-test hooks.
    if (const JsonValue *st = req.at("worker_self_test")) {
        if (st->kind == JsonValue::STR) {
            if (st->s == "crash") worker_crash();
            if (st->s == "hang")  worker_hang();
        }
    }

    // Schema (soft check: warn but proceed — we only trust our own adapter).
    if (const JsonValue *sc = req.at("schema"))
        if (sc->kind != JsonValue::STR || sc->s != kRequestSchema)
            std::fprintf(stderr, "quad_mcf_worker: unexpected schema\n");

    const JsonValue *nodes_j = req.at("nodes");
    const JsonValue *arcs_j  = req.at("arcs");
    const JsonValue *supp_j  = req.at("supply");
    if (!nodes_j || nodes_j->kind != JsonValue::INT || nodes_j->i < 1)
        return json_fail("missing or invalid \"nodes\"");
    if (!arcs_j  || arcs_j->kind  != JsonValue::ARR)
        return json_fail("missing or invalid \"arcs\"");
    if (!supp_j  || supp_j->kind  != JsonValue::ARR)
        return json_fail("missing or invalid \"supply\"");

    const long long N = nodes_j->i;
    if (static_cast<long long>(supp_j->arr.size()) != N)
        return json_fail("supply length != nodes");

    // Validate arc quadruples and build typed LEMON structures.
    const std::size_t A = arcs_j->arr.size();
    std::vector<long long> u_list(A), v_list(A), lo_list(A), up_list(A), c_list(A);
    long long supply_sum = 0;
    for (const JsonValue &x : supp_j->arr) supply_sum += x.i;
    if (supply_sum != 0)
        std::fprintf(stderr, "quad_mcf_worker: note: supply sum != 0 (will be infeasible)\n");

    for (std::size_t k = 0; k < A; ++k) {
        const JsonValue &a = arcs_j->arr[k];
        if (a.kind != JsonValue::ARR || a.arr.size() != 5) {
            char b[64]; snprintf(b, sizeof(b), "arc %zu is not a 5-tuple", k);
            return json_fail(b);
        }
        if (a.arr[0].kind != JsonValue::INT || a.arr[1].kind != JsonValue::INT ||
            a.arr[2].kind != JsonValue::INT || a.arr[3].kind != JsonValue::INT ||
            a.arr[4].kind != JsonValue::INT) {
            char b[64]; snprintf(b, sizeof(b), "arc %zu has non-integer fields", k);
            return json_fail(b);
        }
        const long long u = a.arr[0].i, v = a.arr[1].i;
        const long long lo = a.arr[2].i, up = a.arr[3].i;
        const long long c  = a.arr[4].i;
        if (u < 0 || u >= N || v < 0 || v >= N) {
            char b[96]; snprintf(b, sizeof(b), "arc %zu endpoint out of range", k);
            return json_fail(b);
        }
        if (lo < 0 || up < lo || c < 0) {
            char b[96]; snprintf(b, sizeof(b), "arc %zu bounds/cost invalid", k);
            return json_fail(b);
        }
        u_list[k] = u; v_list[k] = v;
        lo_list[k] = lo; up_list[k] = up; c_list[k] = c;
    }

    // Build a LEMON graph and solve with NetworkSimplex on (long long, long long).
    try {
        lemon::ListDigraph g;
        std::vector<lemon::ListDigraph::Node> node(N);
        for (long long i = 0; i < N; ++i) node[i] = g.addNode();

        std::vector<lemon::ListDigraph::Arc> arc(A);
        for (std::size_t k = 0; k < A; ++k)
            arc[k] = g.addArc(node[u_list[k]], node[v_list[k]]);

        lemon::ListDigraph::NodeMap<long long> supply_map(g);
        for (long long i = 0; i < N; ++i) supply_map[node[i]] = supp_j->arr[i].i;

        lemon::ListDigraph::ArcMap<long long> lower(g);
        lemon::ListDigraph::ArcMap<long long> upper(g);
        lemon::ListDigraph::ArcMap<long long> cost(g);
        for (std::size_t k = 0; k < A; ++k) {
            lower[arc[k]] = lo_list[k];
            upper[arc[k]] = up_list[k];
            cost[arc[k]]  = c_list[k];
        }

        lemon::NetworkSimplex<lemon::ListDigraph, long long, long long> ns(g);
        ns.lowerMap(lower).upperMap(upper).costMap(cost).supplyMap(supply_map);

        lemon::NetworkSimplex<lemon::ListDigraph, long long, long long>::ProblemType res = ns.run();

        std::vector<long long> flows(A);
        long long total = 0;
        if (res == lemon::NetworkSimplex<lemon::ListDigraph, long long, long long>::OPTIMAL) {
            for (std::size_t k = 0; k < A; ++k) flows[k] = ns.flow(arc[k]);
            total = ns.totalCost<long long>();
        }
        const char *status =
            (res == lemon::NetworkSimplex<lemon::ListDigraph, long long, long long>::OPTIMAL)
                ? kStatusOptimal
                : (res == lemon::NetworkSimplex<lemon::ListDigraph, long long, long long>::INFEASIBLE)
                ? kStatusInfeasible
                : kStatusUnbounded;
        write_response(status, total, res == lemon::NetworkSimplex<lemon::ListDigraph, long long, long long>::OPTIMAL ? &flows : nullptr, &g_rc);
        return 0;
    } catch (const std::bad_alloc &) {
        return json_fail("allocation failed");
    } catch (const std::exception &e) {
        return json_fail(e.what());
    } catch (...) {
        return json_fail("unknown solver failure");
    }
}
