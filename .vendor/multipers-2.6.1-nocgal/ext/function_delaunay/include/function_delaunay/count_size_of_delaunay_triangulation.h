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

#include <function_delaunay/Delaunay_triangulation_accessors.h>

#include <gudhi/Simplex_tree.h>


#define DELAUNAY_STATISTICS_DISPATCH(d)					\
    typedef CGAL::Dimension_tag<d> Dim_tag;				\
    typedef Delaunay_triangulation_accessor_d<Dim_tag> DT_accessor;	\
    DT_accessor T(d);							\
    number_of_simplices = _count_size_of_delaunay_triangulation(T,input_points,number_of_k_simplices);

namespace function_delaunay {

    template<typename DelaunayTriangulationAccessor>
    long _count_size_of_delaunay_triangulation(DelaunayTriangulationAccessor& del,
					       std::vector<Point_with_densities> &input_points,
					       std::vector<long> &number_of_k_simplices) {

	typedef DelaunayTriangulationAccessor Delaunay_triangulation_accessor;

	typedef typename Delaunay_triangulation_accessor::Vertex_handle Vertex_handle;
	typedef typename Delaunay_triangulation_accessor::Point Point;
    
#if FUNCTION_DELAUNAY_TIMERS
	test_timer_1.start();
#endif
    
	std::vector<Point> points;
	for(Point_with_densities& point : input_points) { 
	    points.push_back(del.get_point(point));
	}
	del.insert(points.begin(),points.end());
    
	int d = del.current_dimension();

#if FUNCTION_DELAUNAY_TIMERS
	test_timer_1.stop();
#endif
    
	std::cout  << "Delaunay computed, now counting simplices" << std::endl;

	// Compute the size of the Delaunay complex (via simplex tree, as the functionality is not in CGAL)
	int counter=0;
    
	for(auto vit = del.vertices_begin();vit!=del.vertices_end();vit++) {
	    del.data_of_vertex(vit).idx=counter++;
	}
    
	Gudhi::Simplex_tree<> simplex_tree;

	for(auto cit = del.finite_full_cells_begin();cit!=del.finite_full_cells_end();cit++) {
	    std::vector<int> indices;
	    for(int i=0;i<=d;i++) {
		indices.push_back(del.data_of_vertex(cit->vertex(i)).idx);
	    }
	    simplex_tree.insert_simplex_and_subfaces(indices);
	}
	long total_del_size = simplex_tree.num_simplices();
	long skel_size = total_del_size;

	number_of_k_simplices.resize(d+1);
	for(int curr_dim = d-1;curr_dim>=0;curr_dim--) {
	    simplex_tree.prune_above_dimension(curr_dim);
	    long new_size = simplex_tree.num_simplices();
	    number_of_k_simplices[curr_dim+1]=skel_size-new_size;
	    skel_size=new_size;
	}
	number_of_k_simplices[0]=skel_size;

	return total_del_size;

    }



    long count_size_of_delaunay_triangulation(std::vector<Point_with_densities> &input_points,
					      std::vector<long> &number_of_k_simplices) {


	if(input_points.size()==0) {
	    return 0;
	}
	int d = input_points[0].dimension();

	long number_of_simplices=0;

	if(d==2) {
	    // Delaunay_triangulation_2 seems faster than the general one
	    std::cout << "In the plane, using CGAL::Delaunay_triangulation_2" << std::endl;
	    typedef Delaunay_triangulation_accessor_2 DT_accessor;
	    DT_accessor T;
	    number_of_simplices= _count_size_of_delaunay_triangulation(T,input_points,number_of_k_simplices);
	} else if(d==3) {
	    // Same for Delaunay triangulation_3
	    std::cout << "In space, using CGAL::Delaunay_triangulation_3" << std::endl;
	    typedef Delaunay_triangulation_accessor_3 DT_accessor;
	    DT_accessor T;
	    number_of_simplices= _count_size_of_delaunay_triangulation(T,input_points,number_of_k_simplices);
	} else if(d==4) {
	    // Using fixed dimension tags makes the code slightly faster. We dispatch here by
	    // dimension, even if it looks a bit unelegant
	    DELAUNAY_STATISTICS_DISPATCH(4)
		/*
		  } else if(d==5) {
		  DELAUNAY_STATISTICS_DISPATCH(5)      
		  } else if(d==6) {
		  DELAUNAY_STATISTICS_DISPATCH(6)
		  } else if(d==7) {
		  DELAUNAY_STATISTICS_DISPATCH(7)
		  } else if(d==8) {
		  DELAUNAY_STATISTICS_DISPATCH(8)
		  } else if(d==9) {
		  DELAUNAY_STATISTICS_DISPATCH(9)
		  } else if(d==10) {
		  DELAUNAY_STATISTICS_DISPATCH(10)
		*/
		} else {
	    // If you really want to do that, use the general version
	    typedef CGAL::Dynamic_dimension_tag Dim_tag;
	    typedef Delaunay_triangulation_accessor_d<Dim_tag> DT_accessor;
	    DT_accessor T(d);
	    number_of_simplices=_count_size_of_delaunay_triangulation(T,input_points,number_of_k_simplices);
	}
    
	return number_of_simplices;
    }

}
