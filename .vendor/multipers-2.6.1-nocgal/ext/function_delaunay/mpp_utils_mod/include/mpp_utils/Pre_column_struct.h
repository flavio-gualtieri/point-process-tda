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

namespace mpp_utils {

  template<typename Grade_>
  struct Pre_column_struct {
    typedef Grade_ Grade;
    index idx;
    Grade grade;
    std::vector<index> boundary;
    Pre_column_struct() {}
    Pre_column_struct(index idx, Grade& grade, std::vector<index>& boundary)
      : idx(idx), grade(grade), boundary(boundary) {
      std::sort(boundary.begin(),boundary.end());
    }
  };
  

  template<typename Column_,typename Compare>
  struct Sort_pre_column_colex {
    typedef Column_ Column;
    bool operator() (const Column& c1, const Column& c2) {
      Compare comp;
      int n = c1.grade.at.size();
      for(int i=n-1;i>=0;i--) {
	  if(comp(c1.grade.at[i],c2.grade.at[i])) {
	      return true;
	  }
	  if(comp(c2.grade.at[i],c1.grade.at[i])) {
	      return false;
	  }
      }
      return c1.idx<c2.idx;

    }
  };

  template<typename Column_,typename Compare>
  struct Sort_pre_column_lex {
    typedef Column_ Column;
    bool operator() (const Column& c1, const Column& c2) {
      Compare comp;
      int n = c1.grade.at.size();
      for(int i=0;i<n;i++) {
	  if(comp(c1.grade.at[i],c2.grade.at[i])) {
	      return true;
	  }
	  if(comp(c2.grade.at[i],c1.grade.at[i])) {
	      return false;
	  }
      }
      return c1.idx<c2.idx;

    }
      };

 
  template<typename FileParser,typename Pre_column >
  void load_prematrix_contents_from_scc2020_stream(FileParser& parser,
						   int level,
						   std::vector<Pre_column>& pre_matrix) {

    typedef typename FileParser::Grade Coordinate;
    typedef typename FileParser::Field Coefficient;
    typedef typename Pre_column::Grade Grade;
    
    int n = parser.number_of_generators(level);
    //std::cout << "Level: " << level << ", n=" << n << std::endl;
    for(int i=0;i<n;i++) {
      //std::cout << "Info " << level << " " << i << " " << n << std::endl;
      assert(parser.has_next_column(level));
      std::vector<Coordinate> grades;
      std::vector<std::pair<index,Coefficient> > index_pairs;
      parser.next_column(level,
			 std::back_inserter(grades),
			 std::back_inserter(index_pairs));
      assert(grades.size()==2);
      std::vector<index> indices;
      for(auto curr : index_pairs) {
	indices.push_back(curr.first);
      }
      std::sort(indices.begin(),indices.end());
      Grade grade(grades.begin(),grades.end());
      pre_matrix.push_back(Pre_column(i,grade,indices));
    }

  }

  template< typename SimplexTree,typename GradeMap,typename Pre_column >
    void load_prematrix_contents_from_simplex_tree(SimplexTree& simplex_tree,
						   GradeMap& grade,
						   int first_dim,
						   int last_dim,
						   std::vector<std::vector<Pre_column> >& pre_matrices,
						   int& number_of_simplices_in_last_dim) {

    typedef typename Pre_column::Grade Grade;

    pre_matrices.resize(last_dim-first_dim+1);

    std::vector<long> counter_dim;
    for(int i=0;i<=simplex_tree.dimension();i++) {
      counter_dim.push_back(0);
    }
    
    auto all_simplices = simplex_tree.complex_simplex_range();

    for(auto it = all_simplices.begin();it!=all_simplices.end();it++) {
      int curr_dim = simplex_tree.dimension(*it);
      if(curr_dim<first_dim-1 || curr_dim>last_dim) {
	continue;
      }
      simplex_tree.assign_key(*it,counter_dim[curr_dim]);
      counter_dim[curr_dim]++;
    }
    for(auto it = all_simplices.begin();it!=all_simplices.end();it++) {
      int curr_dim = simplex_tree.dimension(*it);
      if(curr_dim<first_dim || curr_dim>last_dim) {
	continue;
      }

      //std::cout << "Simplex has dimension " << simplex_tree.dimension(*it) << std::endl;  
      
      std::vector<index> bd;
      
      for(auto bit=simplex_tree.boundary_simplex_range(*it).begin();
	  bit!=simplex_tree.boundary_simplex_range(*it).end();
	  bit++) {
	bd.push_back(simplex_tree.key(*bit));
      }
      //std::sort(bd.begin(),bd.end());
      
      Grade gr = grade(*it);

      pre_matrices[last_dim-curr_dim].push_back(Pre_column(simplex_tree.key(*it),gr,bd));
    }
    if(first_dim==0) {
      number_of_simplices_in_last_dim=0;
    } else {
      number_of_simplices_in_last_dim=counter_dim[first_dim-1];
    }

  }

}
