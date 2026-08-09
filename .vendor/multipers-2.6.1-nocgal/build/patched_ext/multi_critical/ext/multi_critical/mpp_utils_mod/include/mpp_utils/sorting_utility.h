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

#include<mpp_utils/basic.h>

namespace mpp_utils {

    template<typename Grade>
    bool is_lex_sorted_vector(std::vector<Grade>& grades) {
    
	index n = grades.size();
    
	for(int i=0;i<n-1;i++) {
	    Grade& gr1 = grades[i];
	    Grade& gr2 = grades[i+1];
	    int no_grades = gr1.at.size();
	    for(int p=0;p<no_grades;p++) {
		if(gr1.at[p]<gr2.at[p]) {
		    break;
		}
		if(gr1.at[p]>gr2.at[p]) {
		    return false;
		}
	    }
	}
	return true;
    }


    template<typename GradedMatrix>
    bool is_lex_sorted_columns(GradedMatrix& M) {
    
	return is_lex_sorted_vector(M.grades);
    }

    template<typename GradedMatrix>
    bool is_lex_sorted_rows(GradedMatrix& M) {
    
	return is_lex_sorted_vector(M.row_grades);
    }
  
    template<typename GradedMatrix>
    bool is_lex_sorted(GradedMatrix& M) {
    
	return is_lex_sorted_columns(M) && is_lex_sorted_rows(M);
    }

    template<typename Grade>
    bool is_colex_sorted_vector(std::vector<Grade>& grades) {
    
	index n = grades.size();
    
	for(int i=0;i<n-1;i++) {
	    Grade& gr1 = grades[i];
	    Grade& gr2 = grades[i+1];
	    int no_grades=gr1.at.size();
	    for(int p=no_grades-1;p>=0;p--) {
		if(gr1.at[p]<gr2.at[p]) {
		    break;
		}
		if(gr1.at[p]>gr2.at[p]) {
		    return false;
		}
	    }
	}
	return true;
    }

    template<typename GradedMatrix>
    bool is_colex_sorted_columns(GradedMatrix& M) {
    
	return is_colex_sorted_vector(M.grades);
    }

    template<typename GradedMatrix>
    bool is_colex_sorted_rows(GradedMatrix& M) {
    
	return is_colex_sorted_vector(M.row_grades);
    }

    template<typename GradedMatrix>
    bool is_colex_sorted(GradedMatrix& M) {
    
	return is_colex_sorted_columns(M) && is_colex_sorted_rows(M);
    }

    template<typename GradedMatrix, typename SortingFunctor>
    void reorder_columns(GradedMatrix& M,bool reorder_rows=true,bool do_assign_grade_indices=false) {

	typedef typename SortingFunctor::Column Pre_column;

	SortingFunctor sort_pre_column;

	if(reorder_rows) {

	    // Hack solution: Re-use the Pre_column_struct for this task, even though we have rows here
      
	    std::vector<Pre_column> pre_rows;
	    std::vector<index> empty_col;
	    for(int i=0;i<M.row_grades.size();i++) {
		pre_rows.push_back(Pre_column(i,M.row_grades[i],empty_col));
	    }
	    std::sort(pre_rows.begin(),pre_rows.end(),sort_pre_column);
	    std::vector<index> row_map;
	    row_map.resize(M.num_rows);
	    for(int i=0;i<pre_rows.size();i++) {
		row_map[pre_rows[i].idx]=i;
	    }
	    for(int i=0;i<M.row_grades.size();i++) {
		M.row_grades[i]=pre_rows[i].grade;
	    }
	    // Now iterate through the matrix and update all entries according to row_map
	    for(int i=0;i<M.get_num_cols();i++) {
		std::vector<index> col;
		M.get_col(i,col);
		for(int j=0;j<col.size();j++) {
		    col[j]=row_map[col[j]];
		}
		std::sort(col.begin(),col.end());
		M.set_col(i,col);
	    }
	}

	{ // Now the columns
	    std::vector<Pre_column> pre_cols;
	    for(int i=0;i<M.grades.size();i++) {
		std::vector<index> col;
		M.get_col(i,col);
		pre_cols.push_back(Pre_column(i,M.grades[i],col));
	    }
	    std::sort(pre_cols.begin(),pre_cols.end(),sort_pre_column);
	    for(int i=0;i<M.grades.size();i++) {
		M.grades[i]=pre_cols[i].grade;
		M.set_col(i,pre_cols[i].boundary);
	    }
		  
	}
	if(do_assign_grade_indices) {
	    assign_grade_indices(M);
	}
	
    }
    

    // Re-sorts row and columns of the matrix to have co-lex order
    template<typename GradedMatrix>
    void to_colex_order(GradedMatrix& M,bool reorder_rows=true,bool do_assign_grade_indices=true) {

	typedef typename GradedMatrix::Grade Grade;
	typedef Pre_column_struct<Grade> Pre_column;
	typedef Sort_pre_column_colex<Pre_column,typename GradedMatrix::Coordinate_traits::Compare> SortingFunctor;

	reorder_columns<GradedMatrix,SortingFunctor>(M,reorder_rows,do_assign_grade_indices);
    }

    // Re-sorts row and columns of the matrix to have co-lex order
    template<typename GradedMatrix>
    void to_lex_order(GradedMatrix& M,bool reorder_rows=true,bool do_assign_grade_indices=true) {

	typedef typename GradedMatrix::Grade Grade;
	typedef Pre_column_struct<Grade> Pre_column;
	typedef Sort_pre_column_lex<Pre_column,typename GradedMatrix::Coordinate_traits::Compare> SortingFunctor;

	reorder_columns<GradedMatrix,SortingFunctor>(M,reorder_rows,do_assign_grade_indices);
    }



} // of namespace mpp_utils
