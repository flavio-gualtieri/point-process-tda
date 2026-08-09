/* Copyright 2023 TU Graz
   Author: Michael Kerber
   
   This file is part of function_delaunay
   
   function_delaunay is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   function_delaunay is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with function_delaunay.  If not, see <https://www.gnu.org/licenses/>.
*/


#pragma once

#include <boost/container_hash/hash.hpp>

typedef std::unordered_set<
  std::vector<int>,
  boost::hash<std::vector<int>>
> Uset;

#include <function_delaunay/Point_with_densities.h>
#include <function_delaunay/Delaunay_triangulation_accessors.h>

//#define CGAL_EIGEN3_ENABLED
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Epeck_d.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation.h>

#define GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(d)			\
    typedef CGAL::Dimension_tag<d> Dim_tag;				\
    typedef Delaunay_triangulation_accessor_d<Dim_tag> DT_accessor;	\
    DT_accessor T(d);							\
    _get_simplices_from_triangulation_one_function(T,input_points,simplices);


#ifdef DBG_PRINT
#define PRINT_SIMPLICES(simplices)                                       \
    do {                                                                 \
        std::cout << "Debug: List of simplices at the end of procedure:" << std::endl; \
        for (const auto& s : simplices) {                                \
            for (int v : s) std::cout << v << " ";                       \
            std::cout << std::endl;                                      \
        }                                                                \
        std::cout << "End of final list" << std::endl;                   \
    } while (0)
#else
#define PRINT_SIMPLICES(simplices) do {} while (0)
#endif

namespace function_delaunay {

    template<typename DelaunayTriangulationAccessor> 
    void _get_simplices_from_triangulation_one_function(DelaunayTriangulationAccessor& T ,
							std::vector<Point_with_densities> &input_points,
							std::vector<std::vector<int> > &simplices) {
      
      
	typedef DelaunayTriangulationAccessor Delaunay_triangulation_accessor;
    
	typedef typename Delaunay_triangulation_accessor::Vertex_handle Vertex_handle;
	typedef typename Delaunay_triangulation_accessor::Full_cell_handle Full_cell_handle;
	typedef typename Delaunay_triangulation_accessor::Point Point;

	std::vector<Vertex_handle> vertices_by_idx;
    
	int counter=0;

	int d = input_points[0].dimension();
	
	// Insert the first simplex by hand
	std::vector<int> initial_simplex;
	for(int i=0;i<=d;i++) {
	    initial_simplex.push_back(i);
	}
	simplices.push_back(initial_simplex);

	
	int no_points_outside_convex_hull=0;

	for(Point_with_densities& point : input_points) { 
	    //std::cout << "******************************" << std::endl;
	    //std::cout << "counter=" << counter << std::endl;
	    Point p=T.get_point(point);
      
	    Vertex_handle vh;
            
	    if(T.is_full_dimensional()) {
		int d = T.current_dimension();
		std::vector<Full_cell_handle> cells;
		Full_cell_handle loc = T.locate(p);
		if(T.is_infinite(loc)) {
		    no_points_outside_convex_hull++;
		}
		T.compute_conflict_zone(p,loc,cells);
	
		for(auto cell:cells) {
		    std::vector<int> cell_idx;
		    for(int i=0;i<=d;i++) {
			if(! T.is_infinite(cell->vertex(i))) {
			    cell_idx.push_back(T.data_of_vertex(cell->vertex(i)).idx);
			}
		    }
		    std::sort(cell_idx.begin(),cell_idx.end());
		    cell_idx.push_back(counter);
		    simplices.push_back(cell_idx);
	  
		    /*
		      std::cout << "Cell: ";
		      for(int x: cell_idx) {
		      std::cout <<  x << " ";
		      }
		      std::cout << std::endl;
		    */
		}
		vh = T.insert(p,loc);
	    } else {
		vh = T.insert(p);
	    }
	    T.data_of_vertex(vh).idx = counter;
	    //T.data_of_vertex(vh).density=point.density;
	    vertices_by_idx.push_back(vh);
      
	    counter++;
      
	}
	std::cout << "Number of insertions outside convex hull " << no_points_outside_convex_hull << std::endl;
    }

    void get_simplices_from_triangulation_one_function(std::vector<Point_with_densities> &input_points,
						       std::vector<std::vector<int> > &simplices) {

	if(input_points.size()==0) {
	    return;
	}
	int d=input_points[0].dimension();

	if(d==2) {
	    // Delaunay_triangulation_2 seems faster than the general one
	    std::cout << "In the plane, using CGAL::Delaunay_triangulation_2" << std::endl;
	    typedef Delaunay_triangulation_accessor_2 DT_accessor;
	    DT_accessor T;
	    _get_simplices_from_triangulation_one_function(T,input_points,simplices);
	} else if(d==3) {
	    // Same for Delaunay triangulation_3
	    std::cout << "In space, using CGAL::Delaunay_triangulation_3" << std::endl;
	    typedef Delaunay_triangulation_accessor_3 DT_accessor;
	    DT_accessor T;
	    _get_simplices_from_triangulation_one_function(T,input_points,simplices);
	} else if(d==4) {
	    // Using fixed dimension tags makes the code slightly faster. We dispatch here by
	    // dimension, even if it looks a bit unelegant
	    GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(4)
		} else if(d==5) {
	    GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(5)      
		} else if(d==6) {
	    GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(6)
		} else if(d==7) {
	    GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(7)
		} else if(d==8) {
	    GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(8)
		} else if(d==9) {
	    GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(9)
		} else if(d==10) {
	    GET_SIMPLICES_FROM_TRIANGULATION_ONE_FUNCTION_DISPATCH(10)
		} else {
	    // If you really want to do that, use the general version
	    typedef CGAL::Dynamic_dimension_tag Dim_tag;
	    typedef Delaunay_triangulation_accessor_d<Dim_tag> DT_accessor;
	    DT_accessor T(d);
	    _get_simplices_from_triangulation_one_function(T,input_points,simplices);
	}
    }

    template<typename Delaunay_triangulation_accessor>
    void add_point_to_triangulation(Delaunay_triangulation_accessor& T,
				    Point_with_densities& point,
				    typename Delaunay_triangulation_accessor::Vertex_handle& vh_new,
				    std::vector<std::vector<int>>& conflicts,
				    Uset& simplices,
				    bool& outside_convex_hull,
				    bool fill_conflict_vector=true,
				    bool fill_simplex_vector=true) {

	typedef typename Delaunay_triangulation_accessor::Vertex_handle Vertex_handle;
	typedef typename Delaunay_triangulation_accessor::Full_cell_handle Full_cell_handle;

	typedef typename Delaunay_triangulation_accessor::Point Point;
	Point p=T.get_point(point);

	//std::cout << "Point is " << p[0] << " " << p[1] << std::endl;

	// std::cout << "Is full dimensional " << T.is_full_dimensional() << std::endl;
	// std::cout << "Dimension of T: " << T.current_dimension() << std::endl;

	if(T.is_full_dimensional()) {
	    int d = T.current_dimension();
	    Full_cell_handle loc = T.locate(p);
	    outside_convex_hull=T.is_infinite(loc);
	    std::vector<Full_cell_handle> cells;
	    T.compute_conflict_zone(p,loc,cells);
	    for(auto cell:cells) {
		std::vector<int> cell_idx;
		for(int i=0;i<=d;i++) {
		    if(! T.is_infinite(cell->vertex(i))) {
			cell_idx.push_back(T.data_of_vertex(cell->vertex(i)).idx);
		    }
		}
		std::sort(cell_idx.begin(),cell_idx.end());
		if(fill_conflict_vector) {
		    conflicts.push_back(cell_idx);
		}
		cell_idx.push_back(point.idx);
		std::sort(cell_idx.begin(),cell_idx.end());
		if(fill_simplex_vector) {
		    simplices.insert(cell_idx);
		}

	    }
	    /*
	      std::cout << "Cell: ";
	      for(int x: cell_idx) {
	      std::cout <<  x << " ";
	      }
	      std::cout << std::endl;
	    */
	    vh_new = T.insert(p,loc);
	    T.data_of_vertex(vh_new).idx = point.idx;
	} else {
	    outside_convex_hull=true;
	    //std::cout << "Inserting " << point.idx << ", coors " << point.x[0] << " " << point.x[1] << "..." << std::endl;
	    /*
	      for(auto it = T.vertices_begin(); it!=T.vertices_end();it++) {
	      std::cout << "Present is vertex " << T.data_of_vertex(it).idx << std::endl;
	      }
	    */

	    vh_new = T.insert(p);
	    //std::cout << "done inserting" << std::endl;
	    T.data_of_vertex(vh_new).idx = point.idx;
	    if(fill_simplex_vector && T.is_full_dimensional()) {
		int d = T.current_dimension();
		assert(std::distance(T.finite_full_cells_begin(),T.finite_full_cells_end())==1);
		std::vector<int> cell_idx;
		for(int i=0;i<=d;i++) {
		    cell_idx.push_back(T.data_of_vertex(T.finite_full_cells_begin()->vertex(i)).idx);
		}
		std::sort(cell_idx.begin(),cell_idx.end());
		//if(cell_idx.size()==4) {
		//    std::cout << "Happens here (2)" << std::endl;
		//}
		simplices.insert(cell_idx);
	    }
	}
	//std::cout << "Procedure ends" << std::endl;
    }


    template<typename Delaunay_triangulation_accessor>
    void find_double_conflicts(Delaunay_triangulation_accessor& T,
	                       int idx1, int idx2,
			       std::vector<std::vector<int>> &conflicts1,
			       std::vector<std::vector<int>> &conflicts2,
			       Uset &simplices) {
	
	//std::cout << "Double conflicts: " << idx1 << " " << idx2 << std::endl;

	//std::cout << "Conflict sizes: "  << conflicts1.size() << " " << conflicts2.size() << std::endl;
        
#ifdef DBG_PRINT
	std::cout << "Conflict1:" << std::endl;

	for(auto v : conflicts1) {
	    for (int x : v) {
		std::cout << x << " ";
	    }
	    std::cout << std::endl;
	}

	std::cout << "Conflict2:" << std::endl;

	for(auto v : conflicts2) {
	    for (int x : v) {
		std::cout << x << " ";
	    }
	    std::cout << std::endl;
	}
#endif
	// Quick and dirty solution: Create a std::set with the conflicts1, and just search each element in conflicts2:
	std::set<std::vector<int>> conflict_map1;

	conflict_map1.insert(conflicts1.begin(),conflicts1.end());

	for(auto s : conflicts2) {
	    if(conflict_map1.count(s)) {
		/*
		std::cout << "FOUND DOUBLE CONFLICT: " << std::flush;
		for(int v : s) {
		    std::cout << v << " ";
		}
		std::cout << std::endl;
		*/
		std::vector<int> double_conflict_simplex;
		std::copy(s.begin(),s.end(),std::back_inserter(double_conflict_simplex));
		double_conflict_simplex.push_back(idx1);
		double_conflict_simplex.push_back(idx2);
		std::sort(double_conflict_simplex.begin(),double_conflict_simplex.end());
		simplices.insert(double_conflict_simplex);
		
	    }
	}
	
    }


    template<typename DelaunayTriangulationAccessor>
    void _get_simplices_from_triangulation_two_functions_naive(DelaunayTriangulationAccessor &T,
							 std::vector<Point_with_densities> &input_points,
							 Uset &simplices) {

        std::cout << "Naive routine: \n";

	int outside_convex_hull=0;

	typedef DelaunayTriangulationAccessor Delaunay_triangulation_accessor;
	typedef typename Delaunay_triangulation_accessor::Vertex_handle Vertex_handle;
	typedef typename Delaunay_triangulation_accessor::Full_cell_handle Full_cell_handle;
    
	long number_of_squares_visited=0;
	long number_of_squares_needed=0;
    
	// Sort points wrt 1st coordinate
	std::sort(input_points.begin(),input_points.end(),function_delaunay::Lex_sort_by_density());
	
	// Assign ids for each point
	int counter=0;
	for(auto& p:input_points) {
	    p.idx=counter++;
	}

	// debug printout
	//for(auto& p:input_points) {
	//    std::cout << "ID " << p.idx << ": " << std::flush;
	//    for(int i=0;i<p.x.size();i++) {
	//	std::cout << p.x[i] << " ";
	//    }
	//    std::cout << "("<< p.densities[0] << ", " << p.densities[1] << ")" << std::endl;
	//}

    
	// Idea: Traverse points from left to right, and build the simplices slice by slice vertically
	// This contains the points that we have encoutered so far, sorted by the 2nd grade
	std::vector<Point_with_densities> curr_points;
	std::vector<Point_with_densities> points_to_reinsert;
	std::vector<Vertex_handle> idx_to_vh;
	//idx_to_vh.reserve(input_points.size());
        idx_to_vh.resize(input_points.size());

        //std::vector<Vertex_handle> idx_to_vh(input_points.size());


	for(auto& point:input_points)
        {
            //std::cout << "ID " << point.idx << ": " << std::endl;
            //std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> \n";

	    //std::cout << "****** Handling point " << point.idx << ": " << point.x[0] << " " << point.x[1] << std::endl;
	    //std::cout << "Find position to insert..." << std::flush;
	    // First, remove all points from curr_points from the back
	    while(!curr_points.empty() && curr_points.back().densities[1] > point.densities[1]) {
		Point_with_densities& curr = curr_points.back();
		T.remove(idx_to_vh[curr.idx]);
		points_to_reinsert.push_back(curr);
		curr_points.pop_back();
	    }
	    curr_points.push_back(point);

	    //std::cout << "done, " << points_to_reinsert.size() << " points are above the new point" << std::endl;

	    Vertex_handle vh_new;
	    
	    bool outside_convex_hull;
	    // Next, add the new point
	    //std::cout << "Adding point.." << std::flush;
	    //std::cout << "T has " << std::distance(T.vertices_begin(),T.vertices_end()) << " vertices" << std::endl;

	    // array for conflicts not needed
	    std::vector<std::vector<int>> conflicts_dummy;
	    
	    add_point_to_triangulation(T,point,vh_new,conflicts_dummy,simplices,outside_convex_hull,false,true);

	    // std::cout << "Done, number of simplices: " << simplices.size() << std::endl;
	
	    idx_to_vh[point.idx]=vh_new;
	
	    // Now reinsert the points
	    while(!points_to_reinsert.empty()) {

		long old_number_of_simplices=simplices.size();


		Point_with_densities left = points_to_reinsert.back();
		//std::cout << "Reinserting " << left.idx << " with coordinates " << left.densities[0] << " " << left.densities[1] << std::endl;
		points_to_reinsert.pop_back();

		Vertex_handle vh_left;

		// Remove an re-insert the new point. NOT needed in the first iteration,
		// but in all next ones it seems necessary to get the right conflicts
		T.remove(vh_new);

                //std::cout << "Square dance: right-then-up \n";
		std::vector<std::vector<int>> conflicts_up;
		add_point_to_triangulation(T,point,vh_new,conflicts_up,simplices,outside_convex_hull,true,true);


		//std::cout << "Adding " << left.idx << "..." << std::flush;
		add_point_to_triangulation(T,left,vh_left,conflicts_dummy,simplices,outside_convex_hull,false,true);
		std::cout << "Simplices size at this step: " << simplices.size() << std::endl;

		// Now remove them and re-insert them in opposite order
		//std::cout << "Removing " << point.idx << std::endl;
		//std::cout << "Removing " << left.idx << std::endl;
		T.remove(vh_left);
		T.remove(vh_new);


		std::vector<std::vector<int>> conflicts_right;
		// This step has already been done before, so no simplices need to be added. The crucial info here
		// is the conflicts_right field, to detect double conflicts later
		//std::cout << "readding " << left.idx << std::flush;

                // std::cout << "Square dance: up-then-right \n";
		add_point_to_triangulation(T,left,vh_left,conflicts_right,simplices,outside_convex_hull,true,false);
		//std::cout << "Simplices size: " << simplices.size() << std::endl;
		//std::cout << "Readding " << point.idx << std::flush;
		add_point_to_triangulation(T,point,vh_new,conflicts_dummy,simplices,outside_convex_hull,false,true);
		//std::cout << "Simplices size at this step: " << simplices.size() << std::endl;
		curr_points.push_back(left);
		idx_to_vh[point.idx]=vh_new;
		idx_to_vh[left.idx]=vh_left;

		//std::cout << "Finding double conflicts" << std::endl;
		test_timer_1.resume();
		find_double_conflicts(T,point.idx,left.idx,conflicts_up,conflicts_right,simplices);
		test_timer_1.stop();

                std::cout << "Simplices size at this step: " << simplices.size() << std::endl;
		number_of_squares_visited++;
		if(simplices.size()>old_number_of_simplices) {
		    number_of_squares_needed++;
		}

	    }
	    // std::cout << "At the end of iteration, simplices size: " << simplices.size() << std::endl;

	}

        PRINT_SIMPLICES(simplices);
	std::cout << "Number of points: " << input_points.size() << std::endl;
	std::cout << "Squares visited:  " << number_of_squares_visited << std::endl;
	std::cout << "Squares needed:   " << number_of_squares_needed << std::endl;
    }


    template<typename DelaunayTriangulationAccessor, typename VertexHandle>
    std::unordered_set<int> get_incomparable_vertices_in_neighborhood(DelaunayTriangulationAccessor &T,
                                                        int point_idx, 
                                                        //Uset& simplices,
                                                        std::vector<Point_with_densities>& input_points,
                                                        std::vector<VertexHandle>& idx_to_vh)
        {
        //typedef DelaunayTriangulationAccessor Delaunay_triangulation_accessor;
        //typedef typename Delaunay_triangulation_accessor::Vertex_handle Vertex_handle;
        typedef typename DelaunayTriangulationAccessor::Vertex_handle Vertex_handle;
        //typedef typename Delaunay_triangulation_accessor::Point Point;

        //std::cout << " >>>>> Start get_incomparable_vertices for point id = "
        //          << point_idx << " densities[1] = " << input_points[point_idx].densities[1] << std::endl;

        double base_y = input_points[point_idx].densities[1];
        std::unordered_set<int> higher;
        
        VertexHandle vh = idx_to_vh[point_idx];

        //Point p = T.get_point(input_points[point_idx]);
        //Vertex_handle vh = T.insert(p);

        // no null handle
        if (vh == Vertex_handle()) {
            std::cerr << " idx_to_vh[" << point_idx << "] is nullptr.\n";
            return higher;
        }

        if (T.is_infinite(vh)) {
            std::cerr << " idx_to_vh[" << point_idx << "] is infinite.\n";
            return higher;
        }

        //T.data_of_vertex(vh).idx = point_idx;
        //idx_to_vh[point_idx] = vh;

        // Collect adjacent neighbors
        std::vector<int> neighbor_ids;
        test_timer_3.resume();
        T.get_adjacent_vertex_ids(vh, neighbor_ids);
        test_timer_3.stop();

        // Filter neighbors by density
        for (int v : neighbor_ids) {
            if (input_points[v].densities[1] > base_y) {
                higher.insert(v);
        //        std::cout << v << " (" << input_points[v].densities[1] << ")\n";
            }
        }
        //T.remove(vh);
        //idx_to_vh[point_idx] = Vertex_handle();
        return higher;
    }



//     Experimental alternative
//      Alt routine: conflict-probing + square dance:
//      Iterate: 
//      - Insert p; collect all its incomparable nbhd via vertices of conflict_zones/starzone; 
//      - removem them from triangulation, queue them up for reinserting; remove p; 
//      break when p has no incomparable nbhd
//      Sort the queue decreasingly.
//      BW dance

        template<typename DelaunayTriangulationAccessor>
        void _get_simplices_from_triangulation_two_functions(DelaunayTriangulationAccessor &T,
        							 std::vector<Point_with_densities> &input_points,
        							 Uset &simplices) {

            std::cout << "Calling ALT routine \n";

            typedef DelaunayTriangulationAccessor Delaunay_triangulation_accessor;
            typedef typename Delaunay_triangulation_accessor::Vertex_handle Vertex_handle;
            typedef typename Delaunay_triangulation_accessor::Full_cell_handle Full_cell_handle;

            long number_of_squares_visited=0;
            long number_of_squares_needed=0;

            // Sort points wrt 1st coordinate
            std::sort(input_points.begin(),input_points.end(),function_delaunay::Lex_sort_by_density());

            // Assign ids for each point
            int counter=0;
            for(auto& p:input_points) {
                p.idx=counter++;
            }

            // This contains the points that we have encoutered so far, sorted by the 2nd grade
            std::vector<Point_with_densities> points_to_reinsert;
            std::vector<Vertex_handle> idx_to_vh;
            //idx_to_vh.reserve(input_points.size());
            idx_to_vh.resize(input_points.size());
            //std::vector<Vertex_handle> idx_to_vh(input_points.size());
            for(auto &point : input_points)
            {
                test_timer_2.resume();
                auto p = T.get_point(point);
                Vertex_handle vh = T.insert(p);
                idx_to_vh[point.idx] = vh;
                while (true)
                {
                    // get current incomparable vertices around the pivot point
                    std::unordered_set<int> incomparable_vertices = get_incomparable_vertices_in_neighborhood(T, point.idx, //simplices, 
                                                                                                                input_points, idx_to_vh);
                    if (incomparable_vertices.empty()) {
                        break;
                    }
                    for (int idx : incomparable_vertices) {
                        points_to_reinsert.push_back(input_points[idx]);
                        T.remove(idx_to_vh[idx]);
                        idx_to_vh[idx] = Vertex_handle();
                    }
                    incomparable_vertices.clear();
                }
                if (vh != Vertex_handle()) {
                    T.remove(vh);
                    idx_to_vh[point.idx] = Vertex_handle();
                }
                test_timer_2.stop();
                std::sort(points_to_reinsert.begin(),points_to_reinsert.end(), CoLex_sort_by_density());
                std::reverse(points_to_reinsert.begin(), points_to_reinsert.end());

                // Next, add the new point
                //std::cout << "Adding point.." << std::flush;
                //std::cout << "T has " << std::distance(T.vertices_begin(),T.vertices_end()) << " vertices" << std::endl;
                Vertex_handle vh_new;
                bool outside_convex_hull;

                std::vector<std::vector<int>> conflicts_dummy; // array for conflicts not needed
                add_point_to_triangulation(T,point,vh_new,conflicts_dummy,simplices,outside_convex_hull,false,true);

                // std::cout << "Done, number of simplices: " << simplices.size() << std::endl;
                idx_to_vh[point.idx]=vh_new;

                //std::cout << "Size of points_to_reinsert " << points_to_reinsert.size() << std::endl;

                while(!points_to_reinsert.empty()) {
                    long old_number_of_simplices=simplices.size();
                    Point_with_densities left = points_to_reinsert.back();
                    //std::cout << "Reinserting " << left.idx << " with coordinates (" << left.densities[0] << " " << left.densities[1] << ")" << std::endl;
                    points_to_reinsert.pop_back();

                    Vertex_handle vh_left;

                    // Remove an re-insert the new point. NOT needed in the first iteration,
                    // but in all next ones it seems necessary to get the right conflicts
                    T.remove(vh_new);

                    // Issues start here: when left enters and violates the order of bigrade values
                    // this may produce wrong simplices
                    // dirty fix: collect ones involve point/point.idx, ignore the rest.
                    
                    Uset simplices_dummy;
                    std::vector<std::vector<int>> conflicts_up;
                    //std::vector<std::vector<int>> conflicts_dummy;
                    //std::cout << "Square dance: right-then-up \n";
                    add_point_to_triangulation(T,point,vh_new,conflicts_up,simplices,outside_convex_hull,true,true);

                    //std::cout << "Adding " << left.idx << "..." << std::flush;
                    // >>>> FIX: replace simplices by simplices_dummy
                    add_point_to_triangulation(T,left,vh_left,conflicts_dummy,simplices_dummy,outside_convex_hull,false,true);


                    //std::cout << "Debug: List of simplices dummy at the end of procedure:" << std::endl;
                    for(auto cell : simplices_dummy) {
                        if (std::binary_search(cell.begin(), cell.end(), point.idx)) { 
                            simplices.insert(cell);
                        }
                    }

                    // Now remove them and re-insert them in opposite order
                    T.remove(vh_left);
                    T.remove(vh_new);


                    std::vector<std::vector<int>> conflicts_right;
                    // This step has already been done before, so no simplices need to be added. The crucial info here
                    // is the conflicts_right field, to detect double conflicts later
                    add_point_to_triangulation(T,left,vh_left,conflicts_right,simplices,outside_convex_hull,true,false);
                    add_point_to_triangulation(T,point,vh_new,conflicts_dummy,simplices,outside_convex_hull,false,true);
                    idx_to_vh[point.idx]=vh_new;
                    idx_to_vh[left.idx]=vh_left;

                    //std::cout << "Finding double conflicts" << std::endl;
                    test_timer_1.resume();
                    find_double_conflicts(T,point.idx,left.idx,conflicts_up,conflicts_right,simplices);

                    test_timer_1.stop();
                    number_of_squares_visited++;
                    if(simplices.size()>old_number_of_simplices) {
                        number_of_squares_needed++;
                    }

                }

            }
            PRINT_SIMPLICES(simplices);
            std::cout << "Number of points: " << input_points.size() << std::endl;
            std::cout << "Squares visited:  " << number_of_squares_visited << std::endl;
            std::cout << "Squares needed:   " << number_of_squares_needed << std::endl;	
        }



    void get_simplices_from_triangulation_two_functions(std::vector<Point_with_densities> &input_points,
						        Uset &simplices) {

	if(input_points.size()==0) {
	    return;
	}

	int d=input_points[0].dimension();

	if(d==2) {
	    std::cout << "(two functions) In the plane, using CGAL::Delaunay_triangulation_2" << std::endl;
	    typedef Delaunay_triangulation_accessor_2 Delaunay_triangulation_accessor;
	    Delaunay_triangulation_accessor T;
#ifdef NAIVE
	    _get_simplices_from_triangulation_two_functions_naive(T,input_points,simplices);
#else
	    _get_simplices_from_triangulation_two_functions(T,input_points,simplices);
#endif 
	} else if(d==3) {
	    std::cout << "(two functions) In space, using CGAL::Delaunay_triangulation_3" << std::endl;
      	    typedef Delaunay_triangulation_accessor_3 Delaunay_triangulation_accessor;
	    Delaunay_triangulation_accessor T;
	    //_get_simplices_from_triangulation_two_functions(T,input_points,simplices);
#ifdef NAIVE
	    _get_simplices_from_triangulation_two_functions_naive(T,input_points,simplices);
#else
	    _get_simplices_from_triangulation_two_functions(T,input_points,simplices);
#endif 
	} else {
	    std::cout << "(two functions) In high dim, using CGAL::Delaunay_triangulation_d" << std::endl;
	    typedef CGAL::Dynamic_dimension_tag Dim_tag;
	    typedef Delaunay_triangulation_accessor_d<Dim_tag> Delaunay_triangulation_accessor;
	    Delaunay_triangulation_accessor T(d);
#ifdef NAIVE
	    _get_simplices_from_triangulation_two_functions_naive(T,input_points,simplices);
#else
	    _get_simplices_from_triangulation_two_functions(T,input_points,simplices);
#endif 
	}
    }

    void get_simplices_from_triangulation(std::vector<Point_with_densities> &input_points,
					  std::vector<std::vector<int> > &simplices) {
	if(input_points.size()==0) {
	    return;
	}
	int number_of_functions=input_points[0].number_of_densities();
	if(number_of_functions==1) {
	    get_simplices_from_triangulation_one_function(input_points,simplices);
	} else if(number_of_functions==2) {
	    // TODO: Using hashtables seems not better than just using vectors with duplicates
	    Uset simplices_set;

	    get_simplices_from_triangulation_two_functions(input_points,simplices_set);

	    std::copy(simplices_set.begin(),simplices_set.end(),std::back_inserter(simplices));
	} else {
	    std::cout << "Complex creation currently not implemented for " << number_of_functions << " number of functions" << std::endl;
	}
	return;
    }

}
