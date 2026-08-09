/* Copyright 2021 TU Graz
   Author: Michael Kerber
   
   This file is part of mpp_utils
   
   mpp_utils is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   mpp_utils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with mpp_utils.  If not, see <https://www.gnu.org/licenses/>.*/

#pragma once

#include<cstring>

#include<mpp_utils/basic.h>
#include<mpp_utils/Pre_column_struct.h>
#include<mpp_utils/create_graded_matrices_from_pre_column_struct.h>

namespace mpp_utils {


   
  template<typename SimplexTree,typename GradeMap, typename GradedMatrix>
    void create_graded_matrices_from_simplex_tree(SimplexTree& simplex_tree,
						    GradeMap& grade,
						    int first,
						    int last,
						    std::vector<GradedMatrix>& matrices,
						    bool assign_grade_indices=false) {

    typedef typename GradedMatrix::Grade Grade;
    typedef Pre_column_struct<Grade> Pre_column;
    std::vector<std::vector<Pre_column>> pre_matrices;
    int number_of_simplices_in_last_dim;
    load_prematrix_contents_from_simplex_tree<SimplexTree,GradeMap,Pre_column>
      (simplex_tree,grade,first,last,pre_matrices,number_of_simplices_in_last_dim);
    
    create_graded_matrices_from_pre_column_struct(pre_matrices,
						  matrices,
						  number_of_simplices_in_last_dim,
						  assign_grade_indices);
  }

  // Simplex tree must be Gudhi::Simplex_tree. It is a template argument
  // to avoid the dependency of mpp_utils on gudhi
  // Grade_map is a functor that returns for every simplex in the
  // simplex tree a grade. The value type is expected to be GradedMatrix::Grade
  template<typename SimplexTree,typename GradeMap, typename GradedMatrix>
    void create_graded_matrices_from_simplex_tree(SimplexTree& simplex_tree,
						  GradeMap& grade,
						  std::vector<GradedMatrix>& matrices,
						  bool assign_grade_indices=false) {
    create_graded_matrices_from_simplex_tree(simplex_tree, grade, 0, simplex_tree.dimension(), matrices,assign_grade_indices);
  }

  
}
