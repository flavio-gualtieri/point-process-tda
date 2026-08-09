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

namespace mpp_utils {

  // PreColumn is expected to have type Pre_column_struct<GradedMatrix::Grade>
  // (Template parameter is just for simplfying the types)
  template<typename PreColumn,typename GradedMatrix>
    void create_graded_matrices_from_pre_column_struct(std::vector<std::vector<PreColumn> >& pre_matrices, 
						       std::vector<GradedMatrix>& matrices,
						       int number_of_rows_in_last_matrix=0,
						       bool assign_grade_indices_=false,
						       bool to_colex_order=true) {
    
    typedef PreColumn Pre_column;

    if(to_colex_order) {
	Sort_pre_column_colex<Pre_column,typename GradedMatrix::Coordinate_traits::Compare> sort_pre_column_colex;
	for(int i = 0; i < pre_matrices.size(); i++) {
	    std::sort(pre_matrices[i].begin(),pre_matrices[i].end(),sort_pre_column_colex);
	    if(i>0) {
		std::vector<index> re_index;
		re_index.resize(pre_matrices[i].size());
		for(index j=0;j<pre_matrices[i].size();j++) {
		    re_index[pre_matrices[i][j].idx]=j;
		}
		// I am aware how unreadable this code is! It means: Go through
		// the matrix and replace all boundary indices according to the
		// map just created
		for(index j=0;j<pre_matrices[i-1].size();j++) {
		    for(index k=0;k<pre_matrices[i-1][j].boundary.size();k++) {
			//std::cout << "i-1jk " << i-1 << " " << j << " " << k << std::endl;
			//std::cout << pre_matrices[i-1][j].boundary.size() << std::endl;
			//std::cout << "Renaming " << pre_matrices[i-1][j].boundary[k] << std::endl;
			pre_matrices[i-1][j].boundary[k]=re_index[pre_matrices[i-1][j].boundary[k]];
			//std::cout << "to " << pre_matrices[i-1][j].boundary[k] << std::endl;
		    }
		    std::sort(pre_matrices[i-1][j].boundary.begin(),
			      pre_matrices[i-1][j].boundary.end());
		}
	    }
	}
    }
    
    //std::cout << "Memory after re-indexing " << mem_info() << std::endl;
    matrices.clear();
    matrices.resize(pre_matrices.size());
    for(int i=0; i<pre_matrices.size();i++) {
      int n = pre_matrices[i].size();
      int m = (i<pre_matrices.size()-1) ? pre_matrices[i+1].size() : number_of_rows_in_last_matrix;
      GradedMatrix& matrix=matrices[i];
      matrix.set_dimensions(m,n);
      matrix.grades.resize(n);
      matrix.sync();
#pragma omp parallel for schedule(guided,1)
      for(int j=0;j<n;j++) {
	Pre_column& pcol = pre_matrices[i][j];
        matrix.grades[j]=pcol.grade;
	matrix.set_col(j,pcol.boundary);
      }
      pre_matrices[i].clear();
      pre_matrices[i].shrink_to_fit();
      matrix.sync();
      matrix.num_rows=m;
      if(n>0) {
	  matrix.number_of_parameters=matrix.grades[0].at.size();
      }
    }
    //std::cout << "Memory after assigning matrices " << mem_info() << std::endl;
    if(assign_grade_indices_) {
      assign_grade_indices(matrices.begin(),matrices.end());
    } else {
      copy_grades_to_rows(matrices.begin(),matrices.end());
      for(auto it=matrices.begin();it!=matrices.end();it++) {
	it->grade_indices_assigned=false;
      }
    }
    /* Moved into assign_grade_indices
    for(index i=0;i<matrices.size()-1;i++) {
      for(index j=0;j<matrices[i].num_rows;j++) {
	matrices[i].row_grades.push_back(matrices[i+1].grades[j]);
      }
    }
    */

    //std::cout << "Memory at the end of firep proc " << mem_info() << std::endl;


#if !NDEBUG
    for(index i=0;i<matrices.size()-1;i++) {
      std::cout << "Check sanity " << i << std::endl;
      check_grade_sanity(matrices[i]);
    }
#endif

  }
  
}
