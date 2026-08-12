#pragma once

#include <Python.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace anymesher_native {

using Index = std::int64_t;
using Expansion = std::vector<double>;

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Edge {
    Index first = -1;
    Index second = -1;

    Edge() = default;
    Edge(Index a, Index b)
        : first(std::min(a, b)), second(std::max(a, b)) {}

    bool operator==(const Edge& other) const {
        return first == other.first && second == other.second;
    }

    bool operator<(const Edge& other) const {
        return first < other.first ||
               (first == other.first && second < other.second);
    }
};

struct EdgeHash {
    std::size_t operator()(const Edge& edge) const noexcept {
        const auto first = static_cast<std::uint64_t>(edge.first);
        const auto second = static_cast<std::uint64_t>(edge.second);
        return static_cast<std::size_t>(
            first * UINT64_C(0x9e3779b97f4a7c15) ^
            (second + UINT64_C(0x9e3779b97f4a7c15) + (first << 6) + (first >> 2)));
    }
};

struct Triangle {
    std::array<Index, 3> nodes{};
    bool alive = true;
};

struct Incidence {
    Index first = -1;
    Index second = -1;
};

inline void two_sum(double a, double b, double& x, double& y) {
    x = a + b;
    const double b_virtual = x - a;
    const double a_virtual = x - b_virtual;
    const double b_roundoff = b - b_virtual;
    const double a_roundoff = a - a_virtual;
    y = a_roundoff + b_roundoff;
}

inline void two_diff(double a, double b, double& x, double& y) {
    x = a - b;
    const double b_virtual = a - x;
    const double a_virtual = x + b_virtual;
    const double b_roundoff = b_virtual - b;
    const double a_roundoff = a - a_virtual;
    y = a_roundoff + b_roundoff;
}

inline void split(double value, double& high, double& low) {
    constexpr double splitter = 134217729.0;
    const double product = splitter * value;
    const double abandoned = product - value;
    high = product - abandoned;
    low = value - high;
}

inline void two_product(double a, double b, double& x, double& y) {
    x = a * b;
    double a_high, a_low, b_high, b_low;
    split(a, a_high, a_low);
    split(b, b_high, b_low);
    y = a_low * b_low - (((x - a_high * b_high) - a_low * b_high) - a_high * b_low);
}

inline Expansion grow_expansion(const Expansion& input, double value) {
    Expansion output;
    output.reserve(input.size() + 1);
    double accumulator = value;
    for (double component : input) {
        double sum, tail;
        two_sum(accumulator, component, sum, tail);
        if (tail != 0.0) {
            output.push_back(tail);
        }
        accumulator = sum;
    }
    if (accumulator != 0.0 || output.empty()) {
        output.push_back(accumulator);
    }
    return output;
}

inline Expansion expansion_sum(const Expansion& first, const Expansion& second) {
    Expansion result = first;
    for (double component : second) {
        result = grow_expansion(result, component);
    }
    return result;
}

inline Expansion expansion_negate(const Expansion& input) {
    Expansion result = input;
    for (double& component : result) {
        component = -component;
    }
    return result;
}

inline Expansion expansion_scale(const Expansion& input, double scale) {
    Expansion result;
    for (double component : input) {
        double product, tail;
        two_product(component, scale, product, tail);
        Expansion partial;
        if (tail != 0.0) {
            partial.push_back(tail);
        }
        if (product != 0.0 || partial.empty()) {
            partial.push_back(product);
        }
        result = expansion_sum(result, partial);
    }
    return result;
}

inline Expansion expansion_product(const Expansion& first, const Expansion& second) {
    Expansion result;
    for (double component : second) {
        result = expansion_sum(result, expansion_scale(first, component));
    }
    return result;
}

inline Expansion exact_difference(double first, double second) {
    double difference, tail;
    two_diff(first, second, difference, tail);
    Expansion result;
    if (tail != 0.0) {
        result.push_back(tail);
    }
    if (difference != 0.0 || result.empty()) {
        result.push_back(difference);
    }
    return result;
}

inline int expansion_sign(const Expansion& expansion) {
    for (auto item = expansion.rbegin(); item != expansion.rend(); ++item) {
        if (*item > 0.0) {
            return 1;
        }
        if (*item < 0.0) {
            return -1;
        }
    }
    return 0;
}

inline double expansion_value(const Expansion& expansion) {
    double value = 0.0;
    for (double component : expansion) {
        value += component;
    }
    if (value == 0.0) {
        const int sign = expansion_sign(expansion);
        if (sign != 0) {
            return std::copysign(std::numeric_limits<double>::denorm_min(),
                                 static_cast<double>(sign));
        }
    }
    return value;
}

inline Expansion exact_orient_expansion(
    const Point& first,
    const Point& second,
    const Point& third) {
    const Expansion ax = exact_difference(first.x, third.x);
    const Expansion ay = exact_difference(first.y, third.y);
    const Expansion bx = exact_difference(second.x, third.x);
    const Expansion by = exact_difference(second.y, third.y);
    return expansion_sum(
        expansion_product(ax, by),
        expansion_negate(expansion_product(ay, bx)));
}

inline double robust_orient(
    const Point& first,
    const Point& second,
    const Point& third) {
    const double ax = first.x - third.x;
    const double ay = first.y - third.y;
    const double bx = second.x - third.x;
    const double by = second.y - third.y;
    const double determinant = ax * by - ay * bx;
    const double error = 8.0 * std::numeric_limits<double>::epsilon() *
                         (std::abs(ax * by) + std::abs(ay * bx));
    if (std::abs(determinant) > error) {
        return determinant;
    }
    return expansion_value(exact_orient_expansion(first, second, third));
}

inline Expansion exact_incircle_expansion(
    const Point& first,
    const Point& second,
    const Point& third,
    const Point& point) {
    const Expansion ax = exact_difference(first.x, point.x);
    const Expansion ay = exact_difference(first.y, point.y);
    const Expansion bx = exact_difference(second.x, point.x);
    const Expansion by = exact_difference(second.y, point.y);
    const Expansion cx = exact_difference(third.x, point.x);
    const Expansion cy = exact_difference(third.y, point.y);
    const Expansion alift = expansion_sum(
        expansion_product(ax, ax), expansion_product(ay, ay));
    const Expansion blift = expansion_sum(
        expansion_product(bx, bx), expansion_product(by, by));
    const Expansion clift = expansion_sum(
        expansion_product(cx, cx), expansion_product(cy, cy));
    const Expansion bcdet = expansion_sum(
        expansion_product(bx, cy),
        expansion_negate(expansion_product(by, cx)));
    const Expansion cadet = expansion_sum(
        expansion_product(cx, ay),
        expansion_negate(expansion_product(cy, ax)));
    const Expansion abdet = expansion_sum(
        expansion_product(ax, by),
        expansion_negate(expansion_product(ay, bx)));
    return expansion_sum(
        expansion_sum(expansion_product(alift, bcdet),
                      expansion_product(blift, cadet)),
        expansion_product(clift, abdet));
}

inline double robust_incircle(
    const Point& first,
    const Point& second,
    const Point& third,
    const Point& point) {
    const double ax = first.x - point.x;
    const double ay = first.y - point.y;
    const double bx = second.x - point.x;
    const double by = second.y - point.y;
    const double cx = third.x - point.x;
    const double cy = third.y - point.y;
    const double alift = ax * ax + ay * ay;
    const double blift = bx * bx + by * by;
    const double clift = cx * cx + cy * cy;
    const double first_term = alift * (bx * cy - by * cx);
    const double second_term = blift * (cx * ay - cy * ax);
    const double third_term = clift * (ax * by - ay * bx);
    const double determinant = first_term + second_term + third_term;
    const double scale = std::abs(first_term) + std::abs(second_term) +
                         std::abs(third_term);
    if (std::abs(determinant) >
        32.0 * std::numeric_limits<double>::epsilon() * scale) {
        return determinant;
    }
    return expansion_value(
        exact_incircle_expansion(first, second, third, point));
}

inline bool proper_intersection(
    const Point& a,
    const Point& b,
    const Point& c,
    const Point& d) {
    const double first = robust_orient(a, b, c);
    const double second = robust_orient(a, b, d);
    const double third = robust_orient(c, d, a);
    const double fourth = robust_orient(c, d, b);
    return ((first > 0.0 && second < 0.0) ||
            (first < 0.0 && second > 0.0)) &&
           ((third > 0.0 && fourth < 0.0) ||
            (third < 0.0 && fourth > 0.0));
}

inline bool point_on_segment(
    const Point& point,
    const Point& first,
    const Point& second) {
    if (robust_orient(first, second, point) != 0.0) {
        return false;
    }
    return point.x >= std::min(first.x, second.x) &&
           point.x <= std::max(first.x, second.x) &&
           point.y >= std::min(first.y, second.y) &&
           point.y <= std::max(first.y, second.y);
}

inline bool point_in_ring(
    const Point& point,
    const std::vector<Point>& points,
    const std::vector<Index>& ring) {
    bool inside = false;
    for (std::size_t index = 0; index < ring.size(); ++index) {
        const Point& first = points[static_cast<std::size_t>(ring[index])];
        const Point& second =
            points[static_cast<std::size_t>(ring[(index + 1) % ring.size()])];
        if (point_on_segment(point, first, second)) {
            return true;
        }
        if ((first.y > point.y) != (second.y > point.y)) {
            const double crossing = first.x +
                (point.y - first.y) * (second.x - first.x) /
                (second.y - first.y);
            if (crossing > point.x) {
                inside = !inside;
            }
        }
    }
    return inside;
}

class Triangulator {
public:
    using Cancellation = void (*)(void*, const char*);

    Triangulator(
        std::vector<Point> points,
        std::vector<Edge> segments,
        std::vector<Index> outer,
        std::vector<std::vector<Index>> holes,
        Cancellation cancellation,
        void* cancellation_context)
        : points_(std::move(points)),
          segments_(std::move(segments)),
          outer_(std::move(outer)),
          holes_(std::move(holes)),
          cancellation_(cancellation),
          cancellation_context_(cancellation_context),
          input_count_(static_cast<Index>(points_.size())) {}

    std::vector<std::array<Index, 3>> run() {
        const auto insertion_start = Clock::now();
        add_super_triangle();
        insert_all_points();
        remove_super_triangles();
        insertion_seconds_ = elapsed(insertion_start);

        const auto recovery_start = Clock::now();
        recover_segments();
        recovery_seconds_ = elapsed(recovery_start);

        const auto filter_start = Clock::now();
        auto result = filter_domain();
        filter_seconds_ = elapsed(filter_start);
        workspace_bytes_ = estimate_workspace();
        check_cancel("native triangulation publication");
        return result;
    }

    double insertion_seconds() const { return insertion_seconds_; }
    double recovery_seconds() const { return recovery_seconds_; }
    double filter_seconds() const { return filter_seconds_; }
    std::size_t workspace_bytes() const { return workspace_bytes_; }

private:
    using Clock = std::chrono::steady_clock;

    static double elapsed(const Clock::time_point& start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    void check_cancel(const char* phase) {
        if (cancellation_ != nullptr) {
            cancellation_(cancellation_context_, phase);
        }
    }

    void add_edge(const Edge& edge, Index triangle) {
        auto& incidence = incidence_[edge];
        if (incidence.first < 0) {
            incidence.first = triangle;
        } else if (incidence.second < 0) {
            incidence.second = triangle;
        } else {
            throw std::runtime_error("native triangulation created a nonmanifold edge");
        }
    }

    void remove_edge(const Edge& edge, Index triangle) {
        auto found = incidence_.find(edge);
        if (found == incidence_.end()) {
            throw std::runtime_error("native triangulation lost edge incidence");
        }
        if (found->second.first == triangle) {
            found->second.first = found->second.second;
            found->second.second = -1;
        } else if (found->second.second == triangle) {
            found->second.second = -1;
        } else {
            throw std::runtime_error("native triangulation lost triangle incidence");
        }
        if (found->second.first < 0) {
            incidence_.erase(found);
        }
    }

    Index other_triangle(const Edge& edge, Index triangle) const {
        const auto found = incidence_.find(edge);
        if (found == incidence_.end()) {
            return -1;
        }
        if (found->second.first == triangle) {
            return found->second.second;
        }
        if (found->second.second == triangle) {
            return found->second.first;
        }
        return -1;
    }

    Index add_triangle(Index a, Index b, Index c) {
        const double orientation = robust_orient(
            points_[static_cast<std::size_t>(a)],
            points_[static_cast<std::size_t>(b)],
            points_[static_cast<std::size_t>(c)]);
        if (orientation == 0.0) {
            throw std::runtime_error("native triangulation created a zero-area cell");
        }
        if (orientation < 0.0) {
            std::swap(b, c);
        }
        const Index identifier = static_cast<Index>(triangles_.size());
        triangles_.push_back(Triangle{{a, b, c}, true});
        add_edge(Edge(a, b), identifier);
        add_edge(Edge(b, c), identifier);
        add_edge(Edge(c, a), identifier);
        ++alive_count_;
        return identifier;
    }

    void remove_triangle(Index identifier) {
        Triangle& triangle = triangles_[static_cast<std::size_t>(identifier)];
        if (!triangle.alive) {
            return;
        }
        remove_edge(Edge(triangle.nodes[0], triangle.nodes[1]), identifier);
        remove_edge(Edge(triangle.nodes[1], triangle.nodes[2]), identifier);
        remove_edge(Edge(triangle.nodes[2], triangle.nodes[0]), identifier);
        triangle.alive = false;
        --alive_count_;
    }

    void add_super_triangle() {
        double minimum_x = points_.front().x;
        double maximum_x = points_.front().x;
        double minimum_y = points_.front().y;
        double maximum_y = points_.front().y;
        for (const Point& point : points_) {
            minimum_x = std::min(minimum_x, point.x);
            maximum_x = std::max(maximum_x, point.x);
            minimum_y = std::min(minimum_y, point.y);
            maximum_y = std::max(maximum_y, point.y);
        }
        const double center_x = 0.5 * (minimum_x + maximum_x);
        const double center_y = 0.5 * (minimum_y + maximum_y);
        const double span = std::max({maximum_x - minimum_x,
                                      maximum_y - minimum_y, 1.0});
        points_.push_back(Point{center_x - 32.0 * span, center_y - 16.0 * span});
        points_.push_back(Point{center_x + 32.0 * span, center_y - 16.0 * span});
        points_.push_back(Point{center_x, center_y + 32.0 * span});
        location_seed_ = add_triangle(input_count_, input_count_ + 1, input_count_ + 2);
    }

    Index locate(const Point& point) const {
        Index current = location_seed_;
        if (current < 0 ||
            !triangles_[static_cast<std::size_t>(current)].alive) {
            throw std::runtime_error("native point-location seed is not live");
        }
        const std::size_t limit = incidence_.size() + 16;
        for (std::size_t step = 0; step < limit; ++step) {
            const Triangle& triangle = triangles_[static_cast<std::size_t>(current)];
            bool moved = false;
            for (int local = 0; local < 3; ++local) {
                const Index first = triangle.nodes[local];
                const Index second = triangle.nodes[(local + 1) % 3];
                if (robust_orient(points_[static_cast<std::size_t>(first)],
                                  points_[static_cast<std::size_t>(second)],
                                  point) < 0.0) {
                    const Index next = other_triangle(Edge(first, second), current);
                    if (next < 0) {
                        throw std::runtime_error(
                            "native point location left the super-triangle");
                    }
                    current = next;
                    moved = true;
                    break;
                }
            }
            if (!moved) {
                return current;
            }
        }
        throw std::runtime_error("native point location did not converge");
    }

    void insert_all_points() {
        std::vector<Index> order(static_cast<std::size_t>(input_count_));
        std::iota(order.begin(), order.end(), Index{0});
        std::stable_sort(order.begin(), order.end(), [this](Index first, Index second) {
            const Point& a = points_[static_cast<std::size_t>(first)];
            const Point& b = points_[static_cast<std::size_t>(second)];
            if (a.x != b.x) {
                return a.x < b.x;
            }
            if (a.y != b.y) {
                return a.y < b.y;
            }
            return first < second;
        });

        std::vector<std::uint32_t> visited;
        std::uint32_t stamp = 0;
        for (std::size_t position = 0; position < order.size(); ++position) {
            if (position % 256 == 0) {
                check_cancel("native triangulation insertion");
            }
            const Index point_id = order[position];
            const Point& point = points_[static_cast<std::size_t>(point_id)];
            const Index containing = locate(point);

            if (++stamp == 0) {
                std::fill(visited.begin(), visited.end(), 0);
                stamp = 1;
            }
            if (visited.size() < triangles_.size()) {
                visited.resize(triangles_.size(), 0);
            }
            std::deque<Index> pending{containing};
            std::vector<Index> cavity;
            while (!pending.empty()) {
                const Index identifier = pending.front();
                pending.pop_front();
                if (identifier < 0 ||
                    visited[static_cast<std::size_t>(identifier)] == stamp) {
                    continue;
                }
                visited[static_cast<std::size_t>(identifier)] = stamp;
                const Triangle& triangle =
                    triangles_[static_cast<std::size_t>(identifier)];
                if (!triangle.alive) {
                    continue;
                }
                const double value = robust_incircle(
                    points_[static_cast<std::size_t>(triangle.nodes[0])],
                    points_[static_cast<std::size_t>(triangle.nodes[1])],
                    points_[static_cast<std::size_t>(triangle.nodes[2])],
                    point);
                if (identifier != containing && value <= 0.0) {
                    continue;
                }
                cavity.push_back(identifier);
                for (int local = 0; local < 3; ++local) {
                    const Index neighbor = other_triangle(
                        Edge(triangle.nodes[local],
                             triangle.nodes[(local + 1) % 3]),
                        identifier);
                    if (neighbor >= 0) {
                        pending.push_back(neighbor);
                    }
                }
            }
            if (cavity.empty()) {
                throw std::runtime_error("native Delaunay cavity is empty");
            }

            std::map<Edge, int> boundary_counts;
            for (Index identifier : cavity) {
                const Triangle& triangle =
                    triangles_[static_cast<std::size_t>(identifier)];
                for (int local = 0; local < 3; ++local) {
                    ++boundary_counts[Edge(
                        triangle.nodes[local], triangle.nodes[(local + 1) % 3])];
                }
            }
            for (Index identifier : cavity) {
                remove_triangle(identifier);
            }
            location_seed_ = -1;
            for (const auto& entry : boundary_counts) {
                if (entry.second == 1) {
                    const Index made =
                        add_triangle(entry.first.first, entry.first.second, point_id);
                    if (location_seed_ < 0) {
                        location_seed_ = made;
                    }
                }
            }
            if (location_seed_ < 0) {
                throw std::runtime_error("native Delaunay insertion made no cells");
            }
        }
    }

    void remove_super_triangles() {
        std::vector<Index> remove;
        for (Index identifier = 0;
             identifier < static_cast<Index>(triangles_.size());
             ++identifier) {
            const Triangle& triangle = triangles_[static_cast<std::size_t>(identifier)];
            if (triangle.alive &&
                (triangle.nodes[0] >= input_count_ ||
                 triangle.nodes[1] >= input_count_ ||
                 triangle.nodes[2] >= input_count_)) {
                remove.push_back(identifier);
            }
        }
        for (Index identifier : remove) {
            remove_triangle(identifier);
        }
    }

    Index opposite(const Triangle& triangle, const Edge& edge) const {
        for (Index node : triangle.nodes) {
            if (node != edge.first && node != edge.second) {
                return node;
            }
        }
        return -1;
    }

    bool flip_edge(const Edge& edge) {
        const auto found = incidence_.find(edge);
        if (found == incidence_.end() || found->second.first < 0 ||
            found->second.second < 0) {
            return false;
        }
        const Index first_id = found->second.first;
        const Index second_id = found->second.second;
        const Triangle first = triangles_[static_cast<std::size_t>(first_id)];
        const Triangle second = triangles_[static_cast<std::size_t>(second_id)];
        const Index first_opposite = opposite(first, edge);
        const Index second_opposite = opposite(second, edge);
        if (first_opposite < 0 || second_opposite < 0 ||
            first_opposite == second_opposite) {
            return false;
        }
        const Edge replacement(first_opposite, second_opposite);
        if (incidence_.find(replacement) != incidence_.end()) {
            return false;
        }
        if (!proper_intersection(
                points_[static_cast<std::size_t>(edge.first)],
                points_[static_cast<std::size_t>(edge.second)],
                points_[static_cast<std::size_t>(first_opposite)],
                points_[static_cast<std::size_t>(second_opposite)])) {
            return false;
        }
        remove_triangle(first_id);
        remove_triangle(second_id);
        add_triangle(first_opposite, second_opposite, edge.first);
        add_triangle(second_opposite, first_opposite, edge.second);
        return true;
    }

    void recover_segments() {
        std::sort(segments_.begin(), segments_.end());
        segments_.erase(std::unique(segments_.begin(), segments_.end()),
                        segments_.end());
        std::unordered_set<Edge, EdgeHash> protected_edges;
        for (std::size_t number = 0; number < segments_.size(); ++number) {
            if (number % 64 == 0) {
                check_cancel("native triangulation segment recovery");
            }
            const Edge target = segments_[number];
            const std::size_t limit = std::max<std::size_t>(64, incidence_.size() * 4);
            std::size_t iteration = 0;
            while (incidence_.find(target) == incidence_.end()) {
                if (++iteration > limit) {
                    throw std::runtime_error(
                        "native mandatory-segment recovery did not converge");
                }
                bool found_crossing = false;
                Edge chosen;
                for (const auto& entry : incidence_) {
                    const Edge edge = entry.first;
                    if (edge == target || protected_edges.count(edge) != 0 ||
                        entry.second.first < 0 || entry.second.second < 0 ||
                        edge.first == target.first || edge.first == target.second ||
                        edge.second == target.first || edge.second == target.second) {
                        continue;
                    }
                    if (proper_intersection(
                            points_[static_cast<std::size_t>(target.first)],
                            points_[static_cast<std::size_t>(target.second)],
                            points_[static_cast<std::size_t>(edge.first)],
                            points_[static_cast<std::size_t>(edge.second)]) &&
                        (!found_crossing || edge < chosen)) {
                        chosen = edge;
                        found_crossing = true;
                    }
                }
                if (!found_crossing || !flip_edge(chosen)) {
                    throw std::runtime_error(
                        "native mandatory segment could not be recovered");
                }
            }
            protected_edges.insert(target);
        }
    }

    bool inside_domain(const Point& point) const {
        if (!point_in_ring(point, points_, outer_)) {
            return false;
        }
        for (const auto& hole : holes_) {
            if (point_in_ring(point, points_, hole)) {
                return false;
            }
        }
        return true;
    }

    static std::array<Index, 3> canonical(std::array<Index, 3> triangle) {
        const auto minimum = std::min_element(triangle.begin(), triangle.end());
        std::rotate(triangle.begin(), minimum, triangle.end());
        return triangle;
    }

    std::vector<std::array<Index, 3>> filter_domain() const {
        std::vector<std::array<Index, 3>> result;
        result.reserve(static_cast<std::size_t>(alive_count_));
        for (const Triangle& triangle : triangles_) {
            if (!triangle.alive) {
                continue;
            }
            const Point centroid{
                (points_[static_cast<std::size_t>(triangle.nodes[0])].x +
                 points_[static_cast<std::size_t>(triangle.nodes[1])].x +
                 points_[static_cast<std::size_t>(triangle.nodes[2])].x) / 3.0,
                (points_[static_cast<std::size_t>(triangle.nodes[0])].y +
                 points_[static_cast<std::size_t>(triangle.nodes[1])].y +
                 points_[static_cast<std::size_t>(triangle.nodes[2])].y) / 3.0};
            if (inside_domain(centroid)) {
                result.push_back(canonical(triangle.nodes));
            }
        }
        std::sort(result.begin(), result.end());
        if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
            throw std::runtime_error("native triangulation produced duplicate cells");
        }
        return result;
    }

    std::size_t estimate_workspace() const {
        return points_.capacity() * sizeof(Point) +
               triangles_.capacity() * sizeof(Triangle) +
               incidence_.size() * (sizeof(Edge) + sizeof(Incidence) + 32) +
               segments_.capacity() * sizeof(Edge);
    }

    std::vector<Point> points_;
    std::vector<Edge> segments_;
    std::vector<Index> outer_;
    std::vector<std::vector<Index>> holes_;
    std::vector<Triangle> triangles_;
    std::unordered_map<Edge, Incidence, EdgeHash> incidence_;
    Cancellation cancellation_ = nullptr;
    void* cancellation_context_ = nullptr;
    Index input_count_ = 0;
    Index alive_count_ = 0;
    Index location_seed_ = -1;
    double insertion_seconds_ = 0.0;
    double recovery_seconds_ = 0.0;
    double filter_seconds_ = 0.0;
    std::size_t workspace_bytes_ = 0;
};

struct Buffer {
    Py_buffer view{};
    bool acquired = false;

    ~Buffer() {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }
};

inline bool native_format(const char* format, const char* code) {
    return format != nullptr &&
           (std::strcmp(format, code) == 0 ||
            (format[0] == '@' && std::strcmp(format + 1, code) == 0) ||
            (format[0] == '=' && std::strcmp(format + 1, code) == 0));
}

inline bool acquire_float64_matrix(PyObject* object, Buffer& buffer, const char* name) {
    if (PyObject_GetBuffer(
            object, &buffer.view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        return false;
    }
    buffer.acquired = true;
    if (buffer.view.ndim != 2 || buffer.view.shape == nullptr ||
        buffer.view.shape[1] != 2 || buffer.view.itemsize != 8 ||
        !native_format(buffer.view.format, "d") ||
        !PyBuffer_IsContiguous(&buffer.view, 'C')) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous native float64 matrix with two columns",
                     name);
        return false;
    }
    return true;
}

inline bool acquire_int64_matrix(PyObject* object, Buffer& buffer, const char* name) {
    if (PyObject_GetBuffer(
            object, &buffer.view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        return false;
    }
    buffer.acquired = true;
    if (buffer.view.ndim != 2 || buffer.view.shape == nullptr ||
        buffer.view.shape[1] != 2 || buffer.view.itemsize != 8 ||
        !(native_format(buffer.view.format, "q") ||
          native_format(buffer.view.format, "l")) ||
        !PyBuffer_IsContiguous(&buffer.view, 'C')) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous native signed-int64 matrix with two columns",
                     name);
        return false;
    }
    return true;
}

inline bool acquire_int64_vector(PyObject* object, Buffer& buffer, const char* name) {
    if (PyObject_GetBuffer(
            object, &buffer.view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        return false;
    }
    buffer.acquired = true;
    if (buffer.view.ndim != 1 || buffer.view.shape == nullptr ||
        buffer.view.itemsize != 8 ||
        !(native_format(buffer.view.format, "q") ||
          native_format(buffer.view.format, "l")) ||
        !PyBuffer_IsContiguous(&buffer.view, 'C')) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous native signed-int64 vector",
                     name);
        return false;
    }
    return true;
}

inline double float64_at(const Py_buffer& view, Py_ssize_t row, Py_ssize_t column) {
    const auto* values = static_cast<const double*>(view.buf);
    return values[row * 2 + column];
}

inline Index int64_at(const Py_buffer& view, Py_ssize_t row, Py_ssize_t column = 0) {
    const auto* values = static_cast<const Index*>(view.buf);
    if (view.ndim == 1) {
        return values[row];
    }
    return values[row * 2 + column];
}

struct CallbackContext {
    PyObject* callback = nullptr;
    PyThreadState** released_state = nullptr;
    bool failed = false;
};

struct CallbackFailure {};

inline void invoke_callback(void* raw, const char* phase) {
    auto* context = static_cast<CallbackContext*>(raw);
    if (context == nullptr || context->callback == nullptr ||
        context->callback == Py_None) {
        return;
    }
    PyEval_RestoreThread(*context->released_state);
    *context->released_state = nullptr;
    PyObject* result = PyObject_CallFunction(context->callback, "s", phase);
    if (result == nullptr) {
        context->failed = true;
    } else {
        Py_DECREF(result);
    }
    *context->released_state = PyEval_SaveThread();
    if (context->failed) {
        throw CallbackFailure{};
    }
}

inline PyObject* py_constrained_triangulate(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* segments_object = nullptr;
    PyObject* outer_object = nullptr;
    PyObject* hole_indices_object = nullptr;
    PyObject* hole_offsets_object = nullptr;
    PyObject* callback = Py_None;
    if (!PyArg_ParseTuple(
            args,
            "OOOOO|O:constrained_triangulate",
            &points_object,
            &segments_object,
            &outer_object,
            &hole_indices_object,
            &hole_offsets_object,
            &callback)) {
        return nullptr;
    }
    if (callback != Py_None && !PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "cancellation_check must be callable or None");
        return nullptr;
    }

    Buffer points_buffer;
    Buffer segments_buffer;
    Buffer outer_buffer;
    Buffer hole_indices_buffer;
    Buffer hole_offsets_buffer;
    if (!acquire_float64_matrix(points_object, points_buffer, "points") ||
        !acquire_int64_matrix(segments_object, segments_buffer, "segments") ||
        !acquire_int64_vector(outer_object, outer_buffer, "outer_loop") ||
        !acquire_int64_vector(
            hole_indices_object, hole_indices_buffer, "hole_indices") ||
        !acquire_int64_vector(
            hole_offsets_object, hole_offsets_buffer, "hole_offsets")) {
        return nullptr;
    }
    const Py_ssize_t point_count = points_buffer.view.shape[0];
    if (point_count < 3 || outer_buffer.view.shape[0] < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "compiled triangulation requires at least three points and outer rows");
        return nullptr;
    }

    std::vector<Point> points;
    std::vector<Edge> segments;
    std::vector<Index> outer;
    std::vector<std::vector<Index>> holes;
    try {
        points.reserve(static_cast<std::size_t>(point_count));
        for (Py_ssize_t row = 0; row < point_count; ++row) {
            const Point point{float64_at(points_buffer.view, row, 0),
                              float64_at(points_buffer.view, row, 1)};
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                throw std::runtime_error("compiled triangulation points must be finite");
            }
            points.push_back(point);
        }
        segments.reserve(static_cast<std::size_t>(segments_buffer.view.shape[0]));
        for (Py_ssize_t row = 0; row < segments_buffer.view.shape[0]; ++row) {
            const Index first = int64_at(segments_buffer.view, row, 0);
            const Index second = int64_at(segments_buffer.view, row, 1);
            if (first < 0 || second < 0 || first >= point_count ||
                second >= point_count || first == second) {
                throw std::runtime_error("compiled triangulation received an invalid segment");
            }
            segments.emplace_back(first, second);
        }
        outer.reserve(static_cast<std::size_t>(outer_buffer.view.shape[0]));
        for (Py_ssize_t row = 0; row < outer_buffer.view.shape[0]; ++row) {
            const Index value = int64_at(outer_buffer.view, row);
            if (value < 0 || value >= point_count) {
                throw std::runtime_error("compiled triangulation outer loop is out of range");
            }
            outer.push_back(value);
        }
        const Py_ssize_t offset_count = hole_offsets_buffer.view.shape[0];
        if (offset_count < 1 || int64_at(hole_offsets_buffer.view, 0) != 0 ||
            int64_at(hole_offsets_buffer.view, offset_count - 1) !=
                hole_indices_buffer.view.shape[0]) {
            throw std::runtime_error("compiled triangulation hole offsets are invalid");
        }
        for (Py_ssize_t number = 0; number + 1 < offset_count; ++number) {
            const Index start = int64_at(hole_offsets_buffer.view, number);
            const Index end = int64_at(hole_offsets_buffer.view, number + 1);
            if (start < 0 || end < start || end > hole_indices_buffer.view.shape[0] ||
                end - start < 3) {
                throw std::runtime_error("compiled triangulation hole offsets are invalid");
            }
            std::vector<Index> hole;
            hole.reserve(static_cast<std::size_t>(end - start));
            for (Index row = start; row < end; ++row) {
                const Index value = int64_at(hole_indices_buffer.view, row);
                if (value < 0 || value >= point_count) {
                    throw std::runtime_error("compiled triangulation hole loop is out of range");
                }
                hole.push_back(value);
            }
            holes.push_back(std::move(hole));
        }
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }

    PyThreadState* released_state = PyEval_SaveThread();
    CallbackContext callback_context{callback, &released_state, false};
    std::vector<std::array<Index, 3>> triangles;
    double insertion_seconds = 0.0;
    double recovery_seconds = 0.0;
    double filter_seconds = 0.0;
    std::size_t workspace_bytes = 0;
    try {
        Triangulator triangulator(
            std::move(points),
            std::move(segments),
            std::move(outer),
            std::move(holes),
            invoke_callback,
            &callback_context);
        triangles = triangulator.run();
        insertion_seconds = triangulator.insertion_seconds();
        recovery_seconds = triangulator.recovery_seconds();
        filter_seconds = triangulator.filter_seconds();
        workspace_bytes = triangulator.workspace_bytes();
    } catch (const CallbackFailure&) {
        PyEval_RestoreThread(released_state);
        return nullptr;
    } catch (const std::bad_alloc&) {
        PyEval_RestoreThread(released_state);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyEval_RestoreThread(released_state);
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    PyEval_RestoreThread(released_state);

    PyObject* triangle_list = PyList_New(static_cast<Py_ssize_t>(triangles.size()));
    if (triangle_list == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(triangles.size()); ++row) {
        const auto& triangle = triangles[static_cast<std::size_t>(row)];
        PyObject* value = Py_BuildValue(
            "(LLL)",
            static_cast<long long>(triangle[0]),
            static_cast<long long>(triangle[1]),
            static_cast<long long>(triangle[2]));
        if (value == nullptr) {
            Py_DECREF(triangle_list);
            return nullptr;
        }
        PyList_SET_ITEM(triangle_list, row, value);
    }
    PyObject* diagnostics = Py_BuildValue(
        "{s:d,s:d,s:d,s:K}",
        "insertion_seconds", insertion_seconds,
        "segment_recovery_seconds", recovery_seconds,
        "domain_filter_seconds", filter_seconds,
        "workspace_bytes", static_cast<unsigned long long>(workspace_bytes));
    if (diagnostics == nullptr) {
        Py_DECREF(triangle_list);
        return nullptr;
    }
    return Py_BuildValue("NN", triangle_list, diagnostics);
}

}  // namespace anymesher_native
