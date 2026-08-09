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
#include<mpfree/Grid_scheduler.h>

namespace mpfree {

    template<typename GradedMatrix>
    void chunk_preprocessing(GradedMatrix& M1, GradedMatrix& M2) {
    
	if(verbose) std::cout << "Num entries at start chunk: " << M1.get_num_entries() << std::endl;
	//if(verbose) std::cout << "Num rows 1st matrix: " << M1.num_rows << std::endl;
	//if(verbose) std::cout << "Num cols 2nd matrix: " << M2.get_num_cols() << std::endl;

	std::vector<int> local_pivots;
	for(index i=0;i<M1.num_rows;i++) {
	    local_pivots.push_back(-1);
	}
	if(verbose) std::cout << "Local reduction" << std::endl;
	std::vector<index> global_indices;
	//std::vector<char> local_vec;
	//local_vec.resize(M1.get_num_cols());
	for(index i=0;i<M1.get_num_cols();i++) {
	    while(M1.is_local(i)) {
		index p = M1.get_max_index(i);
		index j = local_pivots[p];
		if(j!=-1) {
		    M1.add_to(j,i);
		} else {
		    local_pivots[p]=i;
		    //local_vec[i]=1;
		    break;
		}
	    }
	    if(!M1.is_local(i)) {
		global_indices.push_back(i);
		//std::cout << i << " is global" << std::endl;
		//local_vec[i]=0;
	    }
	    if(M1.is_empty(i)) {
		// Can save memory
		M1.clear(i);
	    }
	}

	if(verbose) std::cout << "Num entries after local reduce: " << M1.get_num_entries() << std::endl;
    
	if(verbose) std::cout << "Sparsification" << std::endl;
	/*
	  for(index i=0;i<M1.get_num_cols();i++) {
	  if(local_vec[i]) {
	  index p = M1.get_max_index(i);
	  assert(p!=-1);
	  index j = local_pivots[p];
	  assert(j==i);
	  }
	  }
	*/
	std::vector<index> col;
	M1.sync();
#pragma omp parallel for schedule(guided,1), private(col)
	for(index r=0;r<global_indices.size();r++) {
	    col.clear();
	    index i = global_indices[r];
	    //std::cout << "Fetching " << i << std::endl;
	    //assert(!local_vec[i]);
      
	    //std::cout << "i=" << i << std::endl;
	    while(!M1.is_empty(i)) {
		index p = M1.get_max_index(i);
		index j = local_pivots[p];
		if(j!=-1) {
		    //assert(local_vec[j]);
		    //assert(!local_vec[i]);
		    M1.add_to(j,i);
		} else {
		    col.push_back(p);
		    M1.remove_max(i);
		}
	    }
	    std::reverse(col.begin(),col.end());
	    M1.set_col(i,col);
	}
	M1.sync();
	/*
	  for(index i=0;i<M1.get_num_cols();i++) {
	  std::cout << "XXX ";
	  std::vector<index> col;
	  M1.get_col(i,col);
	  for(index j : col) {
	  std::cout << j << " ";
	  }
	  std::cout << std::endl;
	  }
	*/

	if(verbose) std::cout << "Build up smaller matrices" << std::endl;
	std::vector<int> new_row_index;
	new_row_index.resize(M1.num_rows);
	index row_count=0;
	for(index i=0;i<M1.num_rows;i++) {
	    if(local_pivots[i]==-1) {
		new_row_index[i]=row_count++;
	    } else {
		new_row_index[i]=-1;
	    }
	}
	std::vector<int> new_col_index;
	new_col_index.resize(M1.get_num_cols());
	index col_count=0;
	for(index i=0;i<M1.get_num_cols();i++) {
	    if(M1.is_empty(i) || M1.is_local(i)) {
		new_col_index[i]=-1;
	    } else {
		new_col_index[i]=col_count++;
	    }
	}

	//std::cout << "Col count: " << col_count << std::endl;

	for(index i=0;i<M1.get_num_cols();i++) {
	    if(new_col_index[i]!=-1) {
		index j = new_col_index[i];
		assert(j<=i);
		M1.grades[j]=M1.grades[i];
		//std::cout << " Grades: " << M1.grades[i].first_val << " " <<M1.grades[i].second_val << std::endl;
		std::vector<index> col;
		M1.get_col(i,col);
		for(index j=0;j<col.size();j++) {
		    col[j]=new_row_index[col[j]];
		    //std::cout << "Bd: " << new_row_index[col[j]] << std::endl;
		}
		M1.set_col(j,col);
	    }
	}

	M1.set_dimensions(row_count,col_count);
 
    

	for(index i=0;i<M1.num_rows;i++) {
	    if(local_pivots[i]==-1) {
		int j = new_row_index[i];
		assert(j<=i);
		M1.row_grades[j]=M1.row_grades[i];
		M2.grades[j]=M1.row_grades[i];
		std::vector<index> col;
		M2.get_col(i,col);
		M2.set_col(j,col);
	    } 
	}
	M2.set_dimensions(M2.num_rows,row_count);
	M2.grades.resize(row_count);
	M1.row_grades.resize(row_count);
	M1.num_rows=row_count;
    
	M1.assign_slave_matrix();
	M1.assign_pivots();   
	M2.assign_slave_matrix();
	M2.assign_pivots();   


	M1.pq_row.resize(M1.num_grades_y);
	M2.pq_row.resize(M2.num_grades_y);

	if(verbose) std::cout << "After chunk reduction, matrix has " << M1.get_num_cols() << " columns and " << M1.num_rows << " rows" << std::endl;
	if(verbose) std::cout << "N' is " << M1.get_num_cols()+ M1.num_rows << std::endl;

	if(verbose) std::cout << "Num entries after chunk: " << M1.get_num_entries() << std::endl;

	/*
	  M1.print(true,true);
	  M2.print(false,true);
	  std::exit(1);
	*/
    }

    template<typename GradedMatrixInput,typename GradedMatrixOutput>
    void min_gens(GradedMatrixInput& M, GradedMatrixOutput& result, bool use_clearing=false) {
    
	typedef typename GradedMatrixInput::Grade Grade;

	index count=0;

	std::vector<Grade> new_grades;
	std::vector<std::vector<index>> new_cols;

	Grid_scheduler& grid = M.grid_scheduler;

	//int no_grades=0;

	while(! grid.at_end()) {
      
	    /*
	      no_grades++;
	      if(no_grades%10000==0) {
	      std::cout << "no entries: " << M.get_num_entries() << std::endl;
	      std::cout << grid.size() << std::endl;
	      }
	    */
      

	    index_pair new_grade = grid.next_grade();
      
	    index x = new_grade.first;
	    index y = new_grade.second;
	
	    //std::cout << "Min gens " << x << " " << y << std::endl;
	
	    PQ& pq = M.pq_row[y];

	    index_pair range_xy = grid.index_range_at(x,y);
      
	    index start_xy = range_xy.first;
	    index end_xy = range_xy.second;
      
      
	    //std::cout << "Min gens for " << x << " " << y << " traverses through index range " << start_xy << " " << end_xy << std::endl;
	    assert(start_xy<=end_xy);
	    //std::cout << "Before adding, pq of row has size " << pq.size() << std::endl;
	    for(index i = start_xy;i<end_xy;i++) {
		pq.push(i);
	    }
	    //std::cout << "After adding, pq of row has size " << pq.size() << std::endl;
	    /*
	      if(!pq.empty()) {
	      std::cout << "Min gens for " << x << " " << y << " has to do work " << pq.size() << " grade cols: " << end_xy-start_xy << std::endl;

	      }
	    */
	    while(!pq.empty()) {
		index i = pq.top();
		//std::cout << "Index " << i << std::endl;
		// Remove duplicates
		while(!pq.empty() && i==pq.top()) {
		    pq.pop();
		}
		assert(M.grades[i].index_at[0]<=x);
		assert(M.grades[i].index_at[1]==y);
		if(! M.is_empty(i)) {
		    M.reduce_column(i,new_grade,false,true);
		    if(M.is_empty(i)) {
			// Clearing can save memory for some column types
			M.clear(i);
		    }
		    if(!M.is_empty(i) && i>=start_xy&& i<end_xy) {
			std::vector<index> col;
			M.get_col(i,col);
			//std::cout << "NEW MIN GENERATOR Count" << count << " index " << i << " grade " << x << " " << y << std::endl; 
			new_grades.push_back(M.grades[i]);
			new_cols.push_back(col);
		    }
		}
	    }
	}
	result.set_dimensions(M.num_rows,new_cols.size());
	for(index i=0;i<new_cols.size();i++) {
	    result.grades.push_back(new_grades[i]);
	    result.set_col(i,new_cols[i]);
	}
	result.num_rows=M.num_rows;
	result.row_grades=M.row_grades;
	result.number_of_parameters=M.number_of_parameters;

	if(use_clearing) {
	    // check potential of clearing
	    int no_local_pairs=0;
	    std::set<int> columns_saved;
	    int no_cols_with_dominating_pivots=0;
	    for(int i=0;i<result.get_num_cols();i++) {
		if(result.is_empty(i)) {
		    continue;
		}
		if(result.is_local(i)) {
		    index p = result.get_max_index(i);
		    columns_saved.insert(p);
		    no_local_pairs++;
		}
		if(result.pivot_is_dominating(i)) {
		    no_cols_with_dominating_pivots++;
		    index p = result.get_max_index(i);
		    columns_saved.insert(p);
		    result.clearing_info[p]=i;
		}
	
	    }
	    if(verbose) std::cout << "Min gens matrix has " << result.get_num_cols() << " columns, out of which " << no_local_pairs << " are local and " << no_cols_with_dominating_pivots << " have a dominating pivot (" << double(no_cols_with_dominating_pivots*100)/result.get_num_cols() << "%)" << std::endl;
	    if(verbose) std::cout << "Columns replaced: " << columns_saved.size() << std::endl;
	}

    }

    // If no clearing is used, the argument mingens is ignored
    template<typename GradedMatrix>
    void ker_basis(GradedMatrix& M, GradedMatrix& result, GradedMatrix& mingens, bool use_clearing=false) {

	typedef typename GradedMatrix::Grade Grade;
    
	std::vector<Grade> new_grades;
	std::vector<std::vector<index>> new_cols;
    
	std::vector<bool> indices_in_kernel;
	indices_in_kernel.resize(M.get_num_cols());
	for(index i=0;i<M.get_num_cols();i++) {
	    indices_in_kernel[i]=false;
	}

	//std::set<index> indices_in_kernel;

	Grid_scheduler& grid = M.grid_scheduler;
    
    
	while(! grid.at_end()) {
      
	    index_pair new_grade = grid.next_grade();
      
	    index x = new_grade.first;
	    index y = new_grade.second;

	    //std::cout << "At grade " << x << " " << y << std::endl;
      
	    PQ& pq = M.pq_row[y];
      
	    index_pair range_xy = grid.index_range_at(x,y);
      
	    index start_xy = range_xy.first;
	    index end_xy = range_xy.second;

	    assert(start_xy<=end_xy);

	    for(index i = start_xy;i<end_xy;i++) {
		pq.push(i);
	    }
	    //std::cout << "After adding, pq of row has size " << pq.size() << std::endl;
	    while(!pq.empty()) {
		index i = pq.top();
		//std::cout << "next top" << i << std::endl;
		// Remove duplicates
		while(!pq.empty() && i==pq.top()) {
		    pq.pop();
		}
		assert(M.grades[i].index_at[0]<=x);
		assert(M.grades[i].index_at[1]==y);
		if(use_clearing) {
		    if(mingens.clearing_info.count(i)!=0) {
			std::vector<index> col;
			mingens.get_col(mingens.clearing_info[i],col);
			new_cols.push_back(col);
			new_grades.push_back(Grade(x,y,M.x_vals[x],M.y_vals[y]));
			indices_in_kernel[i]=true;;
			//indices_in_kernel.insert(i);
			M.clear(i);
		    }
		}
		M.reduce_column(i,new_grade,true,true);
		if(!indices_in_kernel[i] && M.is_empty(i)) {
		    //if(M.is_empty(i) && indices_in_kernel.count(i)==0) {
		    //std::cout << "NEW KERNEL ELEMENT " << i << " Count: " << count << " Grade " << x << " " << y << std::endl;
		    {
			std::vector<index> col;
			M.slave.get_col(i,col);
			new_cols.push_back(col);
			new_grades.push_back(Grade(x,y,M.x_vals[x],M.y_vals[y]));
	    
			indices_in_kernel[i]=true;
			//indices_in_kernel.insert(i);
			//can save memory
			M.clear(i);
			M.slave.clear(i);
		    }
		}
	    }
	}
	result.number_of_parameters=M.number_of_parameters;
	result.set_dimensions(M.get_num_cols(),new_cols.size());
	for(index i=0;i<new_cols.size();i++) {
	    result.grades.push_back(new_grades[i]);
	    result.set_col(i,new_cols[i]);
	}
	result.num_rows=M.get_num_cols();
	result.slave.set_num_cols(result.get_num_cols());
	result.assign_pivots();
	for(index i=0;i<result.get_num_cols();i++) {
	    result.pivots[result.get_max_index(i)]=i;
	    std::vector<index> slave_col;
	    slave_col.push_back(i);
	    result.slave.set_col(i,slave_col);
	}
	result.num_grades_x=M.num_grades_x;
	result.num_grades_y=M.num_grades_y;
	std::copy(M.x_vals.begin(),M.x_vals.end(),std::back_inserter(result.x_vals));
	std::copy(M.y_vals.begin(),M.y_vals.end(),std::back_inserter(result.y_vals));
    }

    template<typename GradedMatrix>
    void reparameterize(GradedMatrix& cols, GradedMatrix& ker, GradedMatrix& result) {

	index ker_cols = ker.get_num_cols();
	ker.set_dimensions(ker.num_rows,ker_cols + cols.get_num_cols());
	ker.slave.set_dimensions(ker_cols,ker_cols+cols.get_num_cols());
	std::vector<index> empty_vec;
	for(index i = ker_cols;i<ker.get_num_cols();i++) {
	    ker.slave.set_col(i,empty_vec);
	}
	result.set_dimensions(ker_cols,cols.get_num_cols());
    
	/*
	  std::cout << "Pivots:" << std::endl;
	  for(index i=0;i<ker.num_rows;i++) {
	  std::cout << i << " -> " << ker.pivots[i] << std::endl;
	  }
	*/


	std::copy(cols.grades.begin(),cols.grades.end(),std::back_inserter(result.grades));

	index_pair dummy;
	cols.sync();
	ker.sync();
	result.sync();
#pragma omp parallel for schedule(guided,1)
	for(index i=0;i<cols.get_num_cols();i++) {
	    //std::cout << "index" << i << std::endl;
	    std::vector<index> col;
	    cols.get_col(i,col);
	    ker.set_col(ker_cols+i,col);
	    //std::cout << "reduce" << std::endl;
	    ker.reduce_column(ker_cols+i,dummy,true);
	    //std::cout << "done" << std::endl;
	    assert(ker.is_empty(ker_cols+i));
	    std::vector<index> new_col;
	    ker.slave.get_col(ker_cols+i,new_col);
	    result.set_col(i,new_col);
	}
	cols.sync();
	ker.sync();
	result.sync();
	// Assign row grades
	result.num_rows=ker_cols;
	std::copy(ker.grades.begin(),ker.grades.end(),std::back_inserter(result.row_grades));
    
	// We do not trim here yet as it creates trouble
	// in the minimization
	result.num_grades_x=ker.num_grades_x;
	result.num_grades_y=ker.num_grades_y;
	result.number_of_parameters=ker.number_of_parameters;
	std::copy(ker.x_vals.begin(),ker.x_vals.end(),std::back_inserter(result.x_vals));
	std::copy(ker.y_vals.begin(),ker.y_vals.end(),std::back_inserter(result.y_vals));
#if !NDEBUG
	check_grade_sanity(result);
#endif
    }

    template<typename GradedMatrixInput,typename GradedMatrixOutput>
    void minimize(GradedMatrixInput& M,GradedMatrixOutput& result) {

	typedef typename GradedMatrixInput::Grade Grade;

	GradedMatrixInput& VVM=M;

	typedef std::vector<index> Column;


	VVM.assign_pivots();

	std::set<index> rows_to_delete;

	std::vector<index> cols_to_keep;
	std::vector<Grade> col_grades;
	for(index i=0;i<VVM.get_num_cols();i++) {
	    while(! VVM.is_empty(i)) {
	
		index col_grade_x = VVM.grades[i].index_at[0];
		index col_grade_y = VVM.grades[i].index_at[1];
		index p = VVM.get_max_index(i);
		index row_grade_x = VVM.row_grades[p].index_at[0];
		index row_grade_y = VVM.row_grades[p].index_at[1];
	
		//std::cout << "Grades: " << col_grade_x << " " << col_grade_y <<  " -- " <<  row_grade_x << " " << row_grade_y << std::endl;

		//std::cout << "Pivot is " <<  p << std::endl;
		if(col_grade_x!=row_grade_x || col_grade_y!=row_grade_y) {
		    cols_to_keep.push_back(i);
		    col_grades.push_back(VVM.grades[i]);
		    break;
		}
		if(VVM.pivots[p]==-1) {
		    //std::cout << "Found removable pair " << p << "( " <<  VVM.row_grades[p].index_at[0] << " " << VVM.row_grades[p].index_at[1] << " )" << " " << i << "( " << VVM.grades[i].index_at[0] << " " << VVM.grades[i].index_at[1] << " )" << std::endl;
		    rows_to_delete.insert(p);

		    VVM.pivots[p]=i;
		    break;
		} else {
		    //std::cout << "add a column" << VVM.pivots[p] << std::endl;
		    VVM.add_to(VVM.pivots[p],i);
		}
	    }
	    assert(! VVM.is_empty(i));
	}
	std::vector<Column> new_cols;
	new_cols.resize(cols_to_keep.size());


	//std::cout << "HERE I AM " << std::endl;
	VVM.sync();
#pragma omp parallel for schedule(guided,1)
	for(index k=0;k<cols_to_keep.size();k++) {
	    Column& col=new_cols[k];
	    index i = cols_to_keep[k];
	    while(!VVM.is_empty(i)) {
		index p = VVM.get_max_index(i);
		if(VVM.pivots[p]==-1) {
		    col.push_back(p);
		    VVM.remove_max(i);
		} else {
		    VVM.add_to(VVM.pivots[p],i);
		}
	    }
	    std::reverse(col.begin(),col.end());
	}
	VVM.sync();
    
	index nr = VVM.num_rows;
	//std::cout << "Number of rows of VVM: " << nr << std::endl;
	//std::cout << "Have removed " << rows_to_delete.size() << std::endl;
	//std::cout << "Kept " << cols_to_keep.size() << " columns" << std::endl;
	index count=0;
	std::unordered_map<index,index> index_map;
	std::vector<Grade> res_row_grades;
	// Build re-indexing map
	for(index i=0;i<nr;i++) {
	    if(rows_to_delete.count(i)==0) {
		//std::cout << "reindex " << i << " -> " << count << std::endl; 
		index_map[i]=count++;
		res_row_grades.push_back(VVM.row_grades[i]);
	    }
	}
	// Update columns using re-index map
    
	for(index i=0;i<cols_to_keep.size();i++) {
	    //std::cout << "Iterating throgh column " << cols_to_keep[i] << std::endl;
	    Column& col = new_cols[i];
	    for(int j=0;j<col.size();j++) {
		//std::cout << "Lookup " << col[j] << ", count=" << index_map.count(col[j]) << std::endl;
		assert(index_map.count(col[j]));
		col[j]=index_map[col[j]];
	    }
	}
	// Now build up the result
	result.number_of_parameters=M.number_of_parameters;
	result.set_dimensions(index_map.size(),cols_to_keep.size());
	result.num_rows=index_map.size();
	result.grades = col_grades;
	std::copy(res_row_grades.begin(),res_row_grades.end(),std::back_inserter(result.row_grades));
	for(int i=0;i<cols_to_keep.size();i++) {
	    result.set_col(i,new_cols[i]);
	}
	// We should trim the grades here
#if 0
	result.num_grades_x = M.num_grades_x;
	result.num_grades_y = M.num_grades_y;
	std::copy(M.x_vals.begin(),M.x_vals.end(),std::back_inserter(result.x_vals));
	std::copy(M.y_vals.begin(),M.y_vals.end(),std::back_inserter(result.y_vals));
#else
	assign_grade_indices(result);
#endif
    }

} // of namespace mpfree
