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

  template<typename FileParser,typename GradedMatrix>
    void create_graded_matrices_from_scc2020(FileParser& parser, 
					     std::vector<GradedMatrix>& matrices,
					     bool assign_grade_indices=false) {
    create_graded_matrices_from_scc2020(parser, 1, parser.number_of_levels()-1, matrices,assign_grade_indices);
  }

    
  template<typename FileParser,typename GradedMatrix>
    void create_graded_matrices_from_scc2020(FileParser& parser, 
					     int first,
					     int last,
					     std::vector<GradedMatrix>& matrices,
					     bool assign_grade_indices=false) {

    typedef typename GradedMatrix::Grade Grade;
    typedef Pre_column_struct<Grade> Pre_column;
    std::vector<std::vector<Pre_column>> pre_matrices;
    pre_matrices.resize(last-first+1);
    if(verbose) std::cout << "Loading data from file..." << std::flush;
    if(parser.has_grades_on_last_level()) {
      std::cerr << "WARNING: The scc file has row grades on the last level defined\n"
		<< "Currently, these grades are ignored." << std::endl;
    }
    //std::cout << "Memory after file reader " << mem_info() << std::endl;
    for(int lev=first;lev<=last;lev++) {
      load_prematrix_contents_from_scc2020_stream<FileParser,Pre_column>
	(parser,lev,pre_matrices[lev-first]);
    }
    if(verbose) std::cout << "done" << std::endl;
    create_graded_matrices_from_pre_column_struct(pre_matrices,
						  matrices,
						  parser.number_of_generators(last+1),
						  assign_grade_indices);
  }
  
}
