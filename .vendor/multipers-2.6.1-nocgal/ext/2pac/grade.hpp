#ifndef GRADE_H
#define GRADE_H
#include <limits>
#include <algorithm>

/// Represents a bigrade in \f$\mathbf{Z}^2\f$.
struct grade {
    double x, y;
    static constexpr auto min = -std::numeric_limits<decltype(y)>::infinity();
    static constexpr auto max =  std::numeric_limits<decltype(y)>::infinity();
    /// Comparator for lexicographic order on \f$\mathbf{Z}^2\f$.
    struct lex_less {
        bool operator()(const grade& a, const grade& b) const {
            return a.x < b.x || (a.x == b.x && a.y < b.y);
        }
    };
    /// Comparator for lexicographic order on \f$\mathbf{Z}^2\f$.
    struct lex_greater {
        bool operator()(const grade& a, const grade& b) const {
            return a.x > b.x || (a.x == b.x && a.y > b.y);
        }
    };
    /// Comparator for colexicographic order on \f$\mathbf{Z}^2\f$.
    struct colex_less {
        bool operator()(const grade& a, const grade& b) const {
            return a.y < b.y || (a.y == b.y && a.x < b.x);
        }
    };
    /// Comparator for colexicographic order on \f$\mathbf{Z}^2\f$.
    struct colex_greater {
        bool operator()(const grade& a, const grade& b) const {
            return a.y > b.y || (a.y == b.y && a.x > b.x);
        }
    };
    bool operator==(const grade& other) const {
        return x == other.x and y == other.y;
    }
    bool operator!=(const grade& other) const {
        return x != other.x || y != other.y;
    }
    /// Partial order on \f$\mathbf{Z}^2\f$.
    bool operator<=(const grade& other) const {
        return (x <= other.x) && (y <= other.y);
    }
    bool operator>=(const grade& other) const {
        return (x >= other.x) && (y >= other.y);
    }
    bool operator<(const grade& other) const {
        return (x < other.x && y <= other.y) || (x <= other.x && y < other.y);
    }
    /// Join in the lattice \f$\mathbf{Z}^2\f$.
    grade operator|(const grade& other) const {
        return {std::max(x, other.x), std::max(y, other.y)};
    }
    /// Meet in the lattice \f$\mathbf{Z}^2\f$.
    grade operator&(const grade& other) const {
        return {std::min(x, other.x), std::min(y, other.y)};
    }
    grade operator-() const {
        return {-x, -y};
    }
    /// Makes the y-coordinate minimal
    grade to_top() const{
        return {x, min};
    }
    /// Makes the x-coordinate minimal
    grade to_right() const{
        return {min, y};
    }

    operator std::pair<double, double>() const {
        return {x,y};
    }
    bool is_finite(){
        return x > min && x < max && y > min && y < max;
    }
};

#endif
