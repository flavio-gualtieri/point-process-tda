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

#include<mpfree/global.h>

namespace mpfree {

  struct Sort_grades {
    bool operator() (const index_pair& c1, const index_pair& c2) {
      if(c1.first > c2.first) {
	return true;
      }
      if(c1.first < c2.first) {
	return false;
      }
      return c1.second > c2.second;
    }
  };

  class Grid_scheduler {

  public:

    std::priority_queue<index_pair,std::vector<index_pair>,Sort_grades> grades;

    std::map<index_pair,index_pair> index_range;

    index_pair curr_grade;

    Grid_scheduler() {}

    // It is assumed that columns with the same grade appear in M consecutively
    template<typename GradedMatrix>
      Grid_scheduler(GradedMatrix& M) {

      //std::cout << "Grid scheduler with matrix having " << M.num_grades_x << " x-grades and " << M.num_grades_y << " y-grades" << std::endl;
      
      index_pair last_pair=std::make_pair(-1,-1);
      index curr_start=-1;
      for(int i=0;i<M.get_num_cols();i++) {
	index curr_x=M.grades[i].index_at[0];
	index curr_y=M.grades[i].index_at[1];
	assert(curr_x<M.num_grades_x);
	assert(curr_y<M.num_grades_y);
	if(curr_x!=last_pair.first || curr_y !=last_pair.second) {
	  // New grade
	  if(curr_start!=-1) {
	    index_range[last_pair]=std::make_pair(curr_start,i);
	  }
	  curr_start=i;
	  last_pair = std::make_pair(curr_x,curr_y);
	  grades.push(last_pair);
	}
      }
      if(curr_start!=-1) {
	index_range[last_pair]=std::make_pair(curr_start,M.get_num_cols());
      }
      curr_grade=std::make_pair(-1,-1);
	  
    }

    int size() {
      return grades.size();
    }

    bool at_end() {
      return grades.empty();
    }

    index_pair next_grade() {
      index_pair result = grades.top();
      grades.pop();
      while(!grades.empty() && grades.top()==result) {
	grades.pop();
      }
      curr_grade=result;
      return result;
    }

    index_pair index_range_at(index x, index y) {
      auto find_grade = index_range.find(std::make_pair(x,y));
      if(find_grade==index_range.end()) {
	return std::make_pair(0,0);
      }
      return find_grade->second;
    }

    void notify(index x,index y) {
      //std::cout << "Got notified about " << x << " " << y << std::endl;
      if(curr_grade.first!=x || curr_grade.second!=y) {
	grades.push(std::make_pair(x,y));
      }
    }
    
  };


} // of namespace mpfree
