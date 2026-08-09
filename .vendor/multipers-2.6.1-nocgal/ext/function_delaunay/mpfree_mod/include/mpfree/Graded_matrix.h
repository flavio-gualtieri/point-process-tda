/* Copyright 2020 TU Graz
   Author: Michael Kerber
   
   This file is part of mpfree
   
   mpfree is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   mpfree is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with mpfree.  If not, see <https://www.gnu.org/licenses/>.*/

#pragma once

#include<mpp_utils/Graded_matrix.h>

#include<phat/boundary_matrix.h>

#include<algorithm>
#include<unordered_map>
#include <string>

#include <mpfree/global.h>
#include <mpfree/Grid_scheduler.h>


namespace mpfree {

  template<typename GradedMatrixHandle>
    class Graded_matrix_extended : public GradedMatrixHandle {

    public:

    typedef GradedMatrixHandle Graded_matrix_handle;
    
    typedef typename Graded_matrix_handle::Graded_matrix Graded_matrix;
    
    typedef typename Graded_matrix::Representation Representation;

    //Graded_matrix() {}
    Graded_matrix_extended(Graded_matrix *mat) : Graded_matrix_handle(mat) {}
      
    Grid_scheduler grid_scheduler;
    
    std::vector<index> pivots;

    std::vector<PQ> pq_row;

    // Only used if clearing is enabled
    std::unordered_map<index,index> clearing_info;

    phat::boundary_matrix<Representation> slave;

    void reduce_column(index i, index_pair& curr_gr, bool use_slave=false, bool notify_pq=false) {
      

      index p = this->get_max_index(i);

      while(p!=-1 && pivots[p]!=-1 && pivots[p]<i) {
	index k = pivots[p];

	this->add_to(k,i);

	if(use_slave) {
	  slave.add_to(k,i);
	}
	p=this->get_max_index(i);
      }
      if(notify_pq && p!=-1 && pivots[p]>i) {

	index j = pivots[p];
	index gr_y_index = this->grades[j].index_at[1];
	//std::cout << "SCHEDULING COLUMN FOR LATER " << i << " " << j << " " << this->grades[j].index_at[0]<< " " << gr_y_index << std::endl;
	this->pq_row[gr_y_index].push(j);

	index gr_x_index = curr_gr.first;
	this->grid_scheduler.notify(gr_x_index,gr_y_index);
      }
      if(p!=-1 && (pivots[p]==-1 || pivots[p]>i)) {
	pivots[p]=i;
      }
    }

    

    void assign_pivots() {
      pivots.clear();
      for(index i=0;i<this->num_rows;i++) {
	pivots.push_back(-1);
      }
    }

    void assign_slave_matrix() {
      index n = this->get_num_cols();
      slave.set_num_cols(n);
      for(int i=0;i<n;i++) {
	std::vector<index> col;
	col.push_back(i);
	slave.set_col(i,col);
      }
    }

  }; // of class Graded_matrix_extended

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


  
}//of namespace mpfree
