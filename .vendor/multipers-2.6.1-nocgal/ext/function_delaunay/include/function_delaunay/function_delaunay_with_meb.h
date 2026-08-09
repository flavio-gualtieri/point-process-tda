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

#include <mpp_utils/create_graded_matrices_from_simplex_tree.h>
#include <gudhi/Simplex_tree.h>

#include <function_delaunay/Point_with_densities.h>
#include <function_delaunay/compute_meb_radii_of_simplex_tree.h>
#include <function_delaunay/Grade_map.h>
#include <function_delaunay/get_simplices_from_triangulation.h>

namespace function_delaunay {

    // This is the method make_filtration_non_decreasing but with
    // returning the maximal perturbation required.
    // The code was provided by Marc Glisse as a workaround - see
    // https://github.com/GUDHI/gudhi-devel/issues/1269#issuecomment-3508837608
    template<class O> auto adjust_simplex_tree(Gudhi::Simplex_tree<O>& st){
	typedef Gudhi::Simplex_tree<O> ST;
	typedef typename ST::Filtration_value Filtration_value;
	typedef typename ST::Simplex_handle Simplex_handle;
	
	Filtration_value max_fix = 0;
	auto fun = [&](Simplex_handle sh, int dim) -> void {
	    if (dim == 0) return;
	    
	    Filtration_value const& old_filt = st.filtration(sh);
	    Filtration_value new_filt = old_filt;
	    
	    for (Simplex_handle b : st.boundary_simplex_range(sh)) {
		Filtration_value const& b_filt = st.filtration(b);
		if (b_filt > new_filt) new_filt = b_filt;
	    }
	    max_fix = std::max(max_fix, new_filt - old_filt);
	    st.assign_filtration(sh, new_filt);
	};
	st.for_each_simplex(fun);
	
	if (max_fix != 0)
	    st.clear_filtration(); // Drop the cache.
	return max_fix;
    }

    
    // Return the complex as a Gudhi::Simplex_tree instance
    void incremental_delaunay_complex(std::vector<Point_with_densities> &input_points,
				      Gudhi::Simplex_tree<>& simplex_tree,
				      bool only_return_complex_size=false) {

	if(input_points.size()==0) {
	    return;
	}

#if FUNCTION_DELAUNAY_TIMERS
	complex_timer.start();
#endif

	typedef Gudhi::Simplex_tree<> Simplex_tree;
	
	std::cout << "Start complex" << std::endl;
    
	// The simplices are stored just by boundary vertex indices first
	// The filtration values and the order are computed later!
    
	std::vector<std::vector<int> > simplices;

    
	get_simplices_from_triangulation(input_points,simplices);
        
#if FUNCTION_DELAUNAY_TIMERS
	complex_timer.stop();
#endif
    
	std::cout << "Collected " << simplices.size() << " simplices" << std::endl;

    
#if WITH_MEMORY_PROFILE
	std::cout << "Memory after complex: " << mem_info() << std::endl;
#endif

    
#if FUNCTION_DELAUNAY_TIMERS
	face_timer.start();
#endif
  	for(auto simplex : simplices) {
	    simplex_tree.insert_simplex_and_subfaces(simplex);
	}
    
#if FUNCTION_DELAUNAY_TIMERS
	face_timer.stop();
#endif
    
	long total_number_of_simplices = simplex_tree.num_simplices();
    
	std::cout << "Simplex tree has " << simplex_tree.num_vertices() << " vertices and " << total_number_of_simplices << " simplices" << std::endl;
    
#if WITH_MEMORY_PROFILE
	std::cout << "Memory after face: " << mem_info() << std::endl;
#endif
    
	if(only_return_complex_size) {
	    return;
	}

    
#if FUNCTION_DELAUNAY_TIMERS
	meb_timer.start();
#endif

	compute_meb_radii_of_simplex_tree(simplex_tree,input_points);

	std::cout << "Computed all meb values, now sorting" << std::endl;

#if 1 || !NDEBUG || !FUNCTION_DELAUNAY_SMART_MEB_TRAVERSAL

	// This should actually not be necessary, but the numerical issues
	// in meb computation seem to cause problems, in a way that the 
	// meb radius of a simplex can be slightly smaller than for one of
	// its faces
	//bool adjustment_needed = simplex_tree.make_filtration_non_decreasing();
	double adjustment = adjust_simplex_tree(simplex_tree);
	
	if(adjustment>0) {
	    std::cout << "Some meb value had to be adjusted to fit its cofacet" << std::endl;
	    std::cout << "Maximal adjustment: " << adjustment << std::endl;
	} else {
	    std::cout << "No adjustment of meb values was needed" << std::endl;
	}
    
#if FUNCTION_DELAUNAY_SMART_MEB_TRAVERSAL
	//assert(!adjustment_needed);
#endif

#endif



#if FUNCTION_DELAUNAY_TIMERS
	meb_timer.stop();
#endif
    
#if WITH_MEMORY_PROFILE
	std::cout << "Memory after bigrade: " << mem_info() << std::endl;
#endif
    }
	
	

    
    // Returns the total number of simplices
    template<typename GrMat>
    long function_delaunay_with_meb(std::vector<Point_with_densities> &input_points,
				    std::vector<GrMat> &graded_matrices,
				    bool only_return_complex_size=false) {

	if(input_points.size()==0) {
	    return 0;
	}

	typedef Gudhi::Simplex_tree<> Simplex_tree;
	Simplex_tree simplex_tree;

	incremental_delaunay_complex(input_points,simplex_tree,only_return_complex_size);

	long total_number_of_simplices = simplex_tree.num_simplices();	
	
	if(only_return_complex_size) {
	    return total_number_of_simplices;
	}
    
	std::cout << "Building graded boundary matrices" << std::endl;
    
#if FUNCTION_DELAUNAY_TIMERS
	graded_matrices_timer.start();
#endif
    
	Grade_map<Simplex_tree> grade_map(simplex_tree,input_points);
    
	// The "false" is for avoiding to compute the grade indices, which are not needed except for mpfree
	mpp_utils::create_graded_matrices_from_simplex_tree(simplex_tree,grade_map,graded_matrices,false);
    
	int simplex_tree_dimension=simplex_tree.dimension();
    
	// The simplex tree is no longer needed at this point
	// Sort of a hack, but it seems to work
	simplex_tree.prune_above_dimension(-1);
  
	for(int i=0;i<=simplex_tree_dimension;i++) {
	    std::cout << "Simplices in dimension " << i << ": " << graded_matrices[simplex_tree_dimension-i].get_num_cols() << std::endl;
	}
    
#if FUNCTION_DELAUNAY_TIMERS
	graded_matrices_timer.stop();
#endif

#if WITH_MEMORY_PROFILE
	std::cout << "Memory after boundary: " << mem_info() << std::endl;
#endif
  
	return total_number_of_simplices;
    }

  
}
  
