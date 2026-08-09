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

#include <mpp_utils/basic.h>
#include <mpp_utils/Graded_matrix.h>
#include <phat/boundary_matrix.h>

namespace mpp_utils {

    template<typename GradedMatrix>
    class Graded_matrix_handle {
    
    public:

	typedef GradedMatrix Graded_matrix;
    
	typedef typename Graded_matrix::Representation Representation;
	typedef typename Graded_matrix::Coordinate_traits Coordinate_traits;

	typedef typename Graded_matrix::Coordinate Coordinate;
	typedef typename Graded_matrix::Grade Grade;
    
	Graded_matrix *base;

	index& number_of_parameters;
      
	std::vector<Coordinate> &x_vals,&y_vals;

	index &num_grades_x;
	index &num_grades_y;

	index &num_rows;
    
	std::vector<Grade> &grades;

	std::vector<Grade> &row_grades;

	bool &grade_indices_assigned;

	Graded_matrix_handle(Graded_matrix *mat) : base(mat), number_of_parameters(mat->number_of_parameters), x_vals(mat->x_vals), y_vals(mat->y_vals),
						   num_grades_x(mat->num_grades_x), num_grades_y(mat->num_grades_y), num_rows(mat->num_rows),grades(mat->grades), row_grades(mat->row_grades),
						   grade_indices_assigned(mat->grade_indices_assigned) {}

	void print() {
	    base->print();
	}

	bool is_local(index i) {
	    return base->is_local(i);
	}

	bool pivot_is_dominating(index i) {
	    return base->pivot_is_dominating(i);
	}

	template<typename OutStream>
	void print_in_rivet_format(OutStream& out,bool header=true,bool print_rows=true) {
	    base->print_in_rivet_format(out,header,print_rows);
	}

	index get_num_cols() const { return base->get_num_cols(); }

	void set_dimensions( index nr_of_rows, index nr_of_columns ) {base->set_dimensions( nr_of_rows, nr_of_columns );}

	void set_num_cols( index nr_of_columns ) { base->set_dimensions( nr_of_columns, nr_of_columns ); }

	phat::dimension get_dim( index idx ) const { return base->get_dim( idx ); }

	void set_dim( index idx, phat::dimension dim ) { base->set_dim( idx, dim ); }

	void get_col( index idx, phat::column& col  ) const { 
	    base->get_col( idx, col ); 
	}

	void set_col( index idx, const phat::column& col  ) { base->set_col( idx, col ); }

	bool is_empty( index idx ) const { 
	    return base->is_empty(idx); 
	}

	index get_max_index( index idx ) const { 
	    return base->get_max_index(idx);
	}

	void remove_max( index idx ) { 
	    return base->remove_max( idx ); 
	}

	void add_to( index source, index target ) { 
	    base->add_to( source, target ); 
	}

	void clear( index idx ) { 
	    base->clear(idx); 
	}

	void finalize( index idx ) { 
	    base->finalize(idx);
	}
     
	void sync() { base->sync(); }

	phat::dimension get_max_dim() const {
	    return base->get_max_dim();
	}

	index get_num_rows( index idx ) const {
	    return base->get_num_rows(idx);
	}

	index get_max_col_entries() const {
	    return base->get_max_col_entries();
	}

	index get_max_row_entries() const {
	    return base->get_max_row_entries();
	}

	index get_num_entries() const {
	    return base->get_num_entries();
	}

    };
}
