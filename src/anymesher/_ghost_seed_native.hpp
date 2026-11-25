// Native mirror of anymesher._ghost_seed.ghost_bowyer_watson.
//
// Same insertion order, same conflict rules, same start-cell choice (the first
// conflicting ghost cell in sorted order) and the same exact predicate signs
// (ghost_orient / ghost_incircle below: sound float filters, exact
// expansions otherwise, inside the coordinate range the wrapper admits).  The cell set after every
// insertion is a function of those decisions only, so the result equals the
// Python reference.  Any refusal throws std::runtime_error; the caller keeps
// its finite-hull route.
#pragma once

#include "_triangulation_native.hpp"

#include <cfenv>
#include <set>

namespace anymesher_native {

// Exact-signed predicates for the ghost seed.  The filters bound roundoff by
// the permanents (absolute products before subtraction), as the Python
// predicates do; robust_incircle's filter uses the already-cancelled cross
// products and is not used here.  Uncertain signs use exact expansions.
inline double ghost_orient(const Point& a, const Point& b, const Point& c) {
    const double ax = a.x - c.x, ay = a.y - c.y, bx = b.x - c.x, by = b.y - c.y;
    const double left = ax * by, right = ay * bx, determinant = left - right;
    const double error = 8.0 * std::numeric_limits<double>::epsilon() * (std::abs(left) + std::abs(right));
    if (std::abs(determinant) > error) return determinant;
    return static_cast<double>(expansion_sign(exact_orient_expansion(a, b, c)));
}

inline double ghost_incircle(const Point& a, const Point& b, const Point& c, const Point& p) {
    const double ax = a.x - p.x, ay = a.y - p.y, bx = b.x - p.x, by = b.y - p.y;
    const double cx = c.x - p.x, cy = c.y - p.y;
    const double alift = ax * ax + ay * ay, blift = bx * bx + by * by, clift = cx * cx + cy * cy;
    const double determinant = alift * (bx * cy - by * cx) - blift * (ax * cy - ay * cx)
                             + clift * (ax * by - ay * bx);
    const double permanent = alift * (std::abs(bx * cy) + std::abs(by * cx))
                           + blift * (std::abs(ax * cy) + std::abs(ay * cx))
                           + clift * (std::abs(ax * by) + std::abs(ay * bx));
    if (std::abs(determinant) > 64.0 * std::numeric_limits<double>::epsilon() * permanent)
        return determinant;
    return static_cast<double>(expansion_sign(exact_incircle_expansion(a, b, c, p)));
}

// A refusal for a property of the input with a stable code shared with
// _ghost_seed.GhostSeedRefusal (too_few, duplicate, collinear, cocircular,
// rounding_mode).
// Broken invariants throw std::runtime_error: an internal error the Python
// reference retries.
struct GhostRefusal : std::runtime_error {
    GhostRefusal(const char* code_, const char* message) : std::runtime_error(message), code(code_) {}
    const char* code;
};

class GhostSeed {
public:
    using Cancellation = void (*)(void*, const char*);
    static constexpr Index kGhost = -1;

    GhostSeed(std::vector<Point> points, Cancellation cancellation, void* context)
        : points_(std::move(points)), cancellation_(cancellation), context_(context) {}

    std::vector<std::array<Index, 3>> run() {
        const Index count = static_cast<Index>(points_.size());
        if (count < 3) throw GhostRefusal("too_few", "ghost seed needs at least three points");
        std::vector<Index> order(static_cast<std::size_t>(count));
        std::iota(order.begin(), order.end(), Index{0});
        std::sort(order.begin(), order.end(), [this](Index a, Index b) {
            return std::tie(points_[a].x, points_[a].y, a) < std::tie(points_[b].x, points_[b].y, b);
        });
        for (std::size_t i = 1; i < order.size(); ++i) {
            const Point& a = points_[order[i - 1]];
            const Point& b = points_[order[i]];
            if (a.x == b.x && a.y == b.y)
                throw GhostRefusal("duplicate", "ghost seed refuses duplicate stations");
        }
        std::size_t apex = 2;
        while (apex < order.size() &&
               ghost_orient(points_[order[0]], points_[order[1]], points_[order[apex]]) == 0.0)
            ++apex;
        if (apex == order.size()) throw GhostRefusal("collinear", "ghost seed input is collinear");
        const Index top = order[apex];
        const double side = ghost_orient(points_[order[0]], points_[order[1]], points_[top]);
        for (std::size_t i = 0; i + 1 < apex; ++i) {
            const Index u = order[i], v = order[i + 1];
            if (side > 0.0) add_finite(u, v, top);
            else add_finite(v, u, top);
        }
        std::vector<std::pair<Index, Index>> open;
        for (const auto& item : owner_)
            if (owner_.find(DirectedKey{item.first.second, item.first.first}) == owner_.end())
                open.push_back({item.first.first, item.first.second});
        for (const auto& edge : open) add_cell({edge.second, edge.first, kGhost});

        std::vector<Index> cavity, pending;
        std::vector<std::pair<Index, Index>> boundary;
        for (std::size_t position = apex + 1; position < order.size(); ++position) {
            if (position % 64 == 0) check_cancel("native ghost seed insertion");
            const Index p = order[position];
            Index start = -1;
            for (const auto& ghost : ghosts_) {
                const Index id = owner_.at(DirectedKey{ghost.first, ghost.second});
                if (conflict(id, p)) { start = id; break; }
            }
            if (start < 0) throw std::runtime_error("ghost seed could not locate a point");
            ++stamp_;
            if (mark_.size() < cells_.size()) mark_.resize(cells_.size(), 0);
            cavity.clear(); pending.clear(); boundary.clear();
            mark_[start] = stamp_; cavity.push_back(start); pending.push_back(start);
            while (!pending.empty()) {
                const Index id = pending.back(); pending.pop_back();
                const auto cell = cells_[id];
                for (int i = 0; i < 3; ++i) {
                    const Index u = cell[i], v = cell[(i + 1) % 3];
                    const auto found = owner_.find(DirectedKey{v, u});
                    if (found == owner_.end()) throw std::runtime_error("ghost seed lost an edge twin");
                    const Index neighbour = found->second;
                    if (mark_[neighbour] == stamp_) continue;
                    if (conflict(neighbour, p)) {
                        mark_[neighbour] = stamp_;
                        cavity.push_back(neighbour); pending.push_back(neighbour);
                    }
                }
            }
            for (Index id : cavity) {
                const auto cell = cells_[id];
                for (int i = 0; i < 3; ++i) {
                    const Index u = cell[i], v = cell[(i + 1) % 3];
                    if (mark_[owner_.at(DirectedKey{v, u})] != stamp_) boundary.push_back({u, v});
                }
            }
            for (Index id : cavity) remove_cell(id);
            for (const auto& edge : boundary) {
                const Index u = edge.first, v = edge.second;
                if (v == kGhost) add_cell({p, u, kGhost});
                else if (u == kGhost) add_cell({v, p, kGhost});
                else add_finite(u, v, p);
            }
        }

        check_cancel("native ghost seed certificate");
        bool tie = false;
        for (const auto& item : owner_) {
            const Index u = item.first.first, v = item.first.second;
            if (u > v || u == kGhost || v == kGhost) continue;
            const auto& cell = cells_[item.second];
            if (cell[2] == kGhost) continue;
            const auto& other = cells_[owner_.at(DirectedKey{v, u})];
            if (other[2] == kGhost) continue;
            Index opposite = -1;
            for (Index node : other) if (node != u && node != v) opposite = node;
            const double value =
                ghost_incircle(points_[cell[0]], points_[cell[1]], points_[cell[2]], points_[opposite]);
            if (value > 0.0) throw std::runtime_error("ghost seed result is not Delaunay");
            tie = tie || value == 0.0;
        }
        // Exactly cocircular ties admit several Delaunay triangulations; the
        // caller keeps its finite-hull tie choice (see _ghost_seed.py).
        if (tie) throw GhostRefusal("cocircular", "ghost seed has cocircular ties");
        std::vector<std::array<Index, 3>> result;
        for (std::size_t id = 0; id < cells_.size(); ++id) {
            if (!alive_[id] || cells_[id][2] == kGhost) continue;
            auto cell = cells_[id];
            std::rotate(cell.begin(), std::min_element(cell.begin(), cell.end()), cell.end());
            result.push_back(cell);
        }
        std::sort(result.begin(), result.end());
        return result;
    }

private:
    struct DirectedKey {
        Index first, second;
        bool operator==(const DirectedKey& other) const {
            return first == other.first && second == other.second;
        }
    };
    struct DirectedHash {
        std::size_t operator()(const DirectedKey& key) const {
            const std::uint64_t a = static_cast<std::uint64_t>(key.first);
            const std::uint64_t b = static_cast<std::uint64_t>(key.second);
            return static_cast<std::size_t>(a * 0x9E3779B97F4A7C15ULL ^ (b + 0x7F4A7C159E3779B9ULL + (a << 6) + (a >> 2)));
        }
    };

    // The exact predicates assume round-to-nearest.  The seed never changes
    // the floating-point environment; it refuses when the caller's mode is
    // not round-to-nearest (checked on entry) or a callback changed it.
    void check_cancel(const char* phase) {
        if (cancellation_ == nullptr) return;
        cancellation_(context_, phase);
        if (std::fegetround() != FE_TONEAREST)
            throw GhostRefusal("rounding_mode", "a callback changed the rounding mode");
    }

    bool between(Index a, Index b, Index p) const {
        const Point& pa = points_[a]; const Point& pb = points_[b]; const Point& pp = points_[p];
        auto less = [](const Point& x, const Point& y) { return std::tie(x.x, x.y) < std::tie(y.x, y.y); };
        const Point& low = less(pb, pa) ? pb : pa;
        const Point& high = less(pb, pa) ? pa : pb;
        return less(low, pp) && less(pp, high);
    }

    bool conflict(Index id, Index p) const {
        const auto& cell = cells_[id];
        if (cell[2] == kGhost) {
            const double turn = ghost_orient(points_[cell[0]], points_[cell[1]], points_[p]);
            return turn > 0.0 || (turn == 0.0 && between(cell[0], cell[1], p));
        }
        return ghost_incircle(points_[cell[0]], points_[cell[1]], points_[cell[2]], points_[p]) > 0.0;
    }

    void add_finite(Index a, Index b, Index c) {
        if (ghost_orient(points_[a], points_[b], points_[c]) <= 0.0)
            throw std::runtime_error("ghost seed created a non-positive cell");
        add_cell({a, b, c});
    }

    void add_cell(std::array<Index, 3> cell) {
        Index id;
        if (!free_.empty()) { id = free_.back(); free_.pop_back(); cells_[id] = cell; alive_[id] = 1; }
        else { id = static_cast<Index>(cells_.size()); cells_.push_back(cell); alive_.push_back(1); }
        for (int i = 0; i < 3; ++i) {
            if (!owner_.emplace(DirectedKey{cell[i], cell[(i + 1) % 3]}, id).second)
                throw std::runtime_error("ghost seed created a nonmanifold edge");
        }
        if (cell[2] == kGhost) ghosts_.insert({cell[0], cell[1]});
    }

    void remove_cell(Index id) {
        const auto cell = cells_[id];
        for (int i = 0; i < 3; ++i) owner_.erase(DirectedKey{cell[i], cell[(i + 1) % 3]});
        if (cell[2] == kGhost) ghosts_.erase({cell[0], cell[1]});
        alive_[id] = 0;
        free_.push_back(id);
    }

    std::vector<Point> points_;
    Cancellation cancellation_ = nullptr;
    void* context_ = nullptr;
    std::vector<std::array<Index, 3>> cells_;
    std::vector<unsigned char> alive_;
    std::vector<Index> free_;
    std::vector<std::uint32_t> mark_;
    std::uint32_t stamp_ = 0;
    std::unordered_map<DirectedKey, Index, DirectedHash> owner_;
    std::set<std::pair<Index, Index>> ghosts_;
};

inline PyObject* py_ghost_delaunay(PyObject*, PyObject* args) {
    PyObject *points_object = nullptr, *callback = Py_None;
    if (!PyArg_ParseTuple(args, "O|O:ghost_delaunay", &points_object, &callback)) return nullptr;
    if (callback != Py_None && !PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "cancellation must be callable");
        return nullptr;
    }
    Buffer buffer;
    if (!acquire_float64_matrix(points_object, buffer, "points")) return nullptr;
    std::vector<Point> points;
    try {
        points.reserve(static_cast<std::size_t>(buffer.view.shape[0]));
        for (Py_ssize_t row = 0; row < buffer.view.shape[0]; ++row) {
            const Point p{float64_at(buffer.view, row, 0), float64_at(buffer.view, row, 1)};
            for (double v : {p.x, p.y}) {
                if (!std::isfinite(v) || std::abs(v) > std::ldexp(1.0, 64) ||
                    (v != 0.0 && std::abs(v) < std::ldexp(1.0, -64))) {
                    PyErr_SetString(PyExc_ValueError, "ghost seed coordinate range is unsupported");
                    return nullptr;
                }
            }
            points.push_back(p);
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    PyThreadState* released_state = PyEval_SaveThread();
    CallbackContext context{callback, &released_state, false, nullptr, nullptr};
    std::vector<std::array<Index, 3>> triangles;
    if (std::fegetround() != FE_TONEAREST) {
        PyEval_RestoreThread(released_state);
        return Py_BuildValue("(ss)", "rounding_mode", "rounding mode is not round-to-nearest");
    }
    try {
        GhostSeed engine(std::move(points), invoke_callback, &context);
        triangles = engine.run();
    } catch (const CallbackFailure&) {
        PyEval_RestoreThread(released_state);
        return nullptr;
    } catch (const GhostRefusal& refusal) {
        // Refusals are returned, not raised: (code, message).
        PyEval_RestoreThread(released_state);
        return Py_BuildValue("(ss)", refusal.code, refusal.what());
    } catch (const std::bad_alloc&) {
        PyEval_RestoreThread(released_state);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyEval_RestoreThread(released_state);
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    PyEval_RestoreThread(released_state);
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(triangles.size()));
    if (result == nullptr) return nullptr;
    for (std::size_t row = 0; row < triangles.size(); ++row) {
        const auto& t = triangles[row];
        PyObject* value = Py_BuildValue("(LLL)", static_cast<long long>(t[0]),
                                        static_cast<long long>(t[1]), static_cast<long long>(t[2]));
        if (value == nullptr) { Py_DECREF(result); return nullptr; }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(row), value);
    }
    return result;
}

}  // namespace anymesher_native
