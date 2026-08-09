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

#include <mpp_utils/Graded_matrix.h>
#include <function_delaunay/Point_with_densities.h>

namespace function_delaunay {

  // Assign filtration values for simplices in simplex tree
  // Assumes that the meb radius is stored in the simplex tree's filtration value
  // and that the input points are sorted by density
  template<typename SimplexTree>
    class Grade_map {
    
  public: 
    typedef SimplexTree Simplex_tree;
    typedef mpp_utils::Grade_struct<double> Grade;
    
  protected:
    Simplex_tree& simplex_tree;
    std::vector<Point_with_densities>& input_points;
    
  public:
    
    Grade_map(SimplexTree& tree, std::vector<Point_with_densities>& points) 
      : simplex_tree(tree),input_points(points) {
    }
      
      // Returns a grade, as defined in mpp_utils
      template<typename SimplexHandle> Grade operator() (SimplexHandle& handle) {
	
	// We assume that the meb radius has been computed
	// and is stored as filtration value in the simplex
	double meb_radius = simplex_tree.filtration(handle);

	auto it = simplex_tree.simplex_vertex_range(handle).begin();
	assert(it!=simplex_tree.simplex_vertex_range(handle).end());

	std::vector<double> densities;
	std::copy(input_points[*it].densities.begin(),input_points[*it].densities.end(),std::back_inserter(densities));
	it++;
	while(it!=simplex_tree.simplex_vertex_range(handle).end()) {
	    for(int i=0;i<input_points[*it].densities.size();i++) {
		if(input_points[*it].densities[i]>densities[i]) {
		    densities[i]=input_points[*it].densities[i];
		}
	    }
	    it++;
	}
	std::vector<double> grade_results;
	grade_results.push_back(meb_radius);
	std::copy(densities.begin(),densities.end(),std::back_inserter(grade_results));
	
	return Grade(grade_results.begin(),grade_results.end());
	
      }
      
  };
}
