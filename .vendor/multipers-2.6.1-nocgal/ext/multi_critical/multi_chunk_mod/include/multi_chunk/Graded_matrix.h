/* Copyright 2021 TU Graz
   Author: Michael Kerber
   
   This file is part of multi_chunk
   
   multi_chunk is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   multi_chunk is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with multi_chunk.  If not, see <https://www.gnu.org/licenses/>.*/

#pragma once

#include<multi_chunk/basic.h>

#include<mpp_utils/Graded_matrix_handle.h>
#include<mpp_utils/Coordinate_traits_with_map.h>

namespace multi_chunk {

  typedef mpp_utils::index index;
  
  template<typename GradedMatrixHandle>
  class Graded_matrix_extended  : public GradedMatrixHandle {
        
  public:

    typedef GradedMatrixHandle Graded_matrix_handle;
    typedef typename Graded_matrix_handle::Graded_matrix Graded_matrix;

    Graded_matrix_extended(Graded_matrix *mat) : GradedMatrixHandle(mat) {}

    std::vector<index> pivots;
    // 0 = undefined, 1=local positive, -1=local negative, 2=global
    std::vector<char> status;

    std::vector<index> global_index;

    void init() {
      for(index i=0;i<this->get_num_cols();i++) {
	status.push_back(0);
	global_index.push_back(-1);
      }
      for(index i=0;i<this->num_rows;i++) {
	pivots.push_back(-1);
      }
    }
  };

  template<typename GradedMatrix> 
    struct Extend_matrix {
      
      typedef Graded_matrix_extended<mpp_utils::Graded_matrix_handle<GradedMatrix> >Type;
    };
  
  template<typename GradedMatrixHandle>
    struct Extend_matrix<Graded_matrix_extended<GradedMatrixHandle> > {
      typedef Graded_matrix_extended<GradedMatrixHandle> Type;
    };
  
  template<typename GradedMatrix>
    struct Extend_matrix<mpp_utils::Graded_matrix_handle<GradedMatrix> > {
      typedef Graded_matrix_extended<mpp_utils::Graded_matrix_handle<GradedMatrix> > Type;
    };


}
