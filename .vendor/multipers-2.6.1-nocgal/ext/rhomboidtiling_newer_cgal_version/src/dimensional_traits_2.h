#pragma once

#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <vector>

#include "types.h"

template<class K>
class DimensionalTraits_2 {
  public:
    static const int dimension = 2;

    using Point  = typename K::Point_2;
    using Vector = typename K::Vector_2;
    using Sphere = typename K::Circle_2;
    using FT     = typename K::FT;

    using Weight = typename K::FT;
    using Weighted_point = typename K::Weighted_point_2;

    using Vb0 = CGAL::Regular_triangulation_vertex_base_2<K>;
    using Vb  = CGAL::Triangulation_vertex_base_with_info_2<PIndex, K, Vb0>;
    using Fb  = CGAL::Regular_triangulation_face_base_2<K>;
    using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
    using Regular_triangulation = CGAL::Regular_triangulation_2<K, Tds>;
    using Regular_triangulation_finite_cells_iterator = typename Regular_triangulation::Finite_faces_iterator;

    static Regular_triangulation_finite_cells_iterator
    get_finite_cells_begin(const Regular_triangulation& T) {return T.finite_faces_begin();}
    static Regular_triangulation_finite_cells_iterator
    get_finite_cells_end(const Regular_triangulation& T) {return T.finite_faces_end();}

    // Get circumsphere of the points with the given indices.
    static Sphere
    circumsphere(const CVertex& pids, const std::vector<Point>& bpoints) {
        if (pids.size() == 3) {
            return Sphere(bpoints[pids[0]], bpoints[pids[1]], bpoints[pids[2]]); 
        } else if (pids.size() == 2) {
            return Sphere(bpoints[pids[0]], bpoints[pids[1]]); 
        } else {
            // return std::optional<Sphere>() or an exception in the future.
            // Currently there are no cases where this should happen.
            return Sphere(Point(CGAL::ORIGIN));
        }
    }

    // Make a point with the given coordinates.
    static Point
    make_point(const std::vector<double>& coords) {
        if (coords.size() == 2) {
            return Point(coords[0], coords[1]);
        } else {
            // return std::optional<Point>() or an exception in the future.
            return Point(CGAL::ORIGIN);
        }
    }
};
