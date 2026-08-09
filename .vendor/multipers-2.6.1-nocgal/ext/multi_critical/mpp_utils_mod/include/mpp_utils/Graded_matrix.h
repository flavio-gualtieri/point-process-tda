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

#include<phat/boundary_matrix.h>

#include<algorithm>
#include<unordered_map>
#include <string>

#include <mpp_utils/basic.h>
#include <mpp_utils/Coordinate_traits_with_map.h>

#include<mpp_utils/Pre_column_struct.h>
#include<mpp_utils/sorting_utility.h>

namespace mpp_utils {
    
    template<typename Coordinate_>
    struct Grade_struct {
	typedef Coordinate_ Coordinate;
	std::vector<Coordinate> at;
	std::vector<index> index_at;
	Grade_struct() {}
	//Constructors for bigrade
	Grade_struct(Coordinate x, Coordinate y) {
	    at.push_back(x);
	    at.push_back(y);
	}
	Grade_struct(index ind_x, index ind_y, Coordinate val_x, Coordinate val_y) {
	    at.push_back(val_x);
	    at.push_back(val_y);
	    index_at.push_back(ind_x);
	    index_at.push_back(ind_y);
	}
	//Constructur for general grade
	template<typename InputIterator>
	Grade_struct(InputIterator gr_begin, InputIterator gr_end) {
	    std::copy(gr_begin,gr_end,std::back_inserter(at));
	}
	
	Grade_struct(const Grade_struct<Coordinate>& other) {
	    std::copy(other.at.begin(),other.at.end(),std::back_inserter(this->at));
	    std::copy(other.index_at.begin(),other.index_at.end(),std::back_inserter(this->index_at));
	}
	
	bool operator== (const Grade_struct<Coordinate>& other) {
	    int n=this->at.size();
	    if(other.at.size()!=n) {
		return false;
	    }
	    for(int i=0;i<n;i++) {
		if(this->at[i]!=other.at[i]) {
		    return false;
		}
	    }
	    return true;
	}
    };
    
    template<typename Representation_=phat::vector_vector, 
	     typename CoordinateTraits = Coordinate_traits_with_map<double> >
    class Graded_matrix  : public phat::boundary_matrix<Representation_> {
    
    public:

	typedef Representation_ Representation;
	typedef CoordinateTraits Coordinate_traits;
    
	typedef typename Coordinate_traits::Coordinate Coordinate;

	typedef Grade_struct<Coordinate> Grade;

	typedef phat::index index;

	index number_of_parameters;

	/*
	 * The following values are only used in mpfree and require
	 * exactly two parameters. They are filled using the function
	 * assign_grade_indices which is only used for mpfree.
	 * TODO: Move this into mpfree
	 */
	std::vector<Coordinate> x_vals,y_vals;

	index num_grades_x;
	index num_grades_y;
	// end of mpfree-specific values
	
	index num_rows;
    
	std::vector<Grade> grades;

	std::vector<Grade> row_grades;

	bool grade_indices_assigned;

	/*
	  Graded_matrix() {}

	  Graded_matrix(Graded_matrix& other) {
	  std::copy(other.x_vals.begin(),other.x_vals.end(),std::back_inserter(this->x_vals));
	  std::copy(other.y_vals.begin(),other.y_vals.end(),std::back_inserter(this->y_vals));
	  // not complete
	  }
	*/
    
	void print(bool print_row_grades=false,bool print_indices=true) {
	    std::cout << "Number of parameters: " << this->number_of_parameters << std::endl;
	    std::cout << "Number of columns: " << this->get_num_cols() << std::endl;
	    std::cout << "Number of rows: " << this->num_rows << std::endl;

	    for(int i=0;i<this->get_num_cols();i++) {
		if(print_indices) {
		    for(int p=0;p<number_of_parameters;p++) {
			std::cout << grades[i].index_at[p] <<  " ";
		    }
		    std::cout << " -- ";
		} else {
		    for(int p=0;p<number_of_parameters;p++) {
			std::cout << grades[i].at[p] <<  " ";
		    }
		    std::cout << " -- ";
		}
		phat::column col;
		this->get_col(i,col);
		for(std::size_t j=0;j<col.size();j++) {
		    std::cout << col[j] << " ";
		}
		std::cout << std::endl;

	    }

	    if(print_row_grades) {
		std::cout << "Row grades:" << std::endl;
		for(index j=0;j<this->num_rows;j++) {
		    std::cout << j << " : ";
		    if(print_indices) {
			for(int p=0;p<number_of_parameters;p++) {
			    std::cout << row_grades[j].index_at[p] <<  " ";
			}
			std::cout << " -- ";
		    } else {
			for(int p=0;p<number_of_parameters;p++) {
			    std::cout << row_grades[j].at[p] <<  " ";
			}
			std::cout << " -- ";
		    }
		}
	    }
      
	}

	bool is_local(index i) {
	    index p = this->get_max_index(i);
	    //std::cout << "Info: " << i << " "<<  p  << "rows " << this->row_grades.size() << std::endl;
	    return p!=-1 && (this->grades[i] == this->row_grades[p]);
      
	}

	bool pivot_is_dominating(index i) {
	    if(this->is_empty(i)) {
		return false;
	    }
	    if(this->is_local(i)) {
		return true;
	    }
	    index p = this->get_max_index(i);
	    Grade& pgr = this->row_grades[p];
	    //std::cout << "#### NEW COLUMN ####" << std::endl;
	    //std::cout << "pivot grade is " << pgr.first_index << ", " << pgr.second_index << std::endl;
	    std::vector<index> col;
	    this->get_col(i,col);
	    for(int j=0;j<col.size();j++) {
		Grade& curr=this->row_grades[col[j]];

		for(int i=0;i<number_of_parameters;i++) {
		    if(curr.at[i]>pgr.at[i]) {
			//std::cout << "NOT HERE" << std::endl;
			return false;
		    }
		}
	    }
      
	    return true;
	}


	template<typename OutStream>
	void print_in_rivet_format(OutStream& out,bool header=true,bool print_rows=true) {

	    // Relict: header should only be used if the number of parameters is 2
	    if(header) {
		out << "firep\nfirst parameter\nsecond parameter\n";

		out << this->get_num_cols() << " " << this->num_rows << " 0" << std::endl;
	    }	

	    Coordinate_traits& traits = Coordinate_traits::get_instance();

	    for(index i=0;i<this->get_num_cols();i++) {
		for(int p=0;p<this->number_of_parameters;p++) {
		    traits.to_stream(out,grades[i].at[p]);
		    out <<  " ";
		}
		out << "; ";
		std::vector<index> col;
		this->get_col(i,col);
		for(index j=0;j<col.size();j++) {
		    out << col[j] << " ";
		}
		out << "\n";
	    }
	    if(print_rows) {
		//std::cout << "Printing rows " << num_rows << " " << row_grades.size() << std::endl;
		for(index i=0;i<num_rows;i++) {
		    for(int p=0;p<number_of_parameters;p++) {
			traits.to_stream(out,row_grades[i].at[p]);
			out <<  " ";
		    }
		    out << "; \n";
		}
	    }
	    out << std::flush;
	}

	// Careful: This is very inefficient...don't use it in code that
	// should be fast. Adds row a of the matrix to row b
	void row_add_to(index a, index b) {
	    int n = this->get_num_cols();
	    for(int i=0;i<n;i++) {
		std::vector<index> col;
		this->get_col(i,col);
		// Do linear search - even slower than binary search!
		if(std::find(col.begin(),col.end(),a)!=col.end()) {
		    auto pos_of_b = std::find(col.begin(),col.end(),b);
		    if(pos_of_b==col.end()) {
			col.push_back(b);
			std::sort(col.begin(),col.end());
		    } else {
			col.erase(pos_of_b);
		    }
		    this->set_col(i,col);
		}
	    }
	}
		    
	
	// Returns whether the grade at column a is smaller than the grade at column b (in all coordinates)
	bool grade_smaller(index a,index b) {
	    int p = number_of_parameters;
	    for(int i=0;i<p;i++) {
		if(grades[a].at[i]>grades[b].at[i]) {
		    return false;
		}
	    }
	    return true;
	}

	bool row_grade_smaller(index a,index b) {
	    int p = number_of_parameters;
	    for(int i=0;i<p;i++) {
		if(row_grades[a].at[i]>row_grades[b].at[i]) {
		    return false;
		}
	    }
	    return true;
	}

	int shuffle_columns() {

	    int n = this->get_num_cols();

	    if(n<=1) {
		return 0;
	    }
	    
	    int source=0;
	    int target=0;
	    
	    while(source==target) {
		source = rand()%n;
		target = rand()%n;
	    }
	    if(grade_smaller(target,source)) {
		std::swap(source,target);
	    }
	    if(grade_smaller(source,target)) {
		this->add_to(source,target);
		return true;
	    }
	    return false;
	}

	int shuffle_rows() {

	    int n = this->num_rows;

	    if(n<=1) {
		return 0;
	    }
	    
	    int source=0;
	    int target=0;
	    
	    while(source==target) {
		source = rand()%n;
		target = rand()%n;
	    }
	    if(row_grade_smaller(source,target)) {
		std::swap(source,target);
	    }
	    if(row_grade_smaller(target,source)) {
		this->row_add_to(source,target);
		return true;
	    }
	    return false;
	}
	    
	int shuffle_columns(int no_attempts) {

	    int number_of_additions=0;

	    for(int i=0;i<no_attempts;i++) {

		if(shuffle_columns()) {
		    number_of_additions++;

		}
		
	    }
	    return number_of_additions;
	}

	int shuffle_rows_and_columns(int no_attempts,double ratio_of_col=0.5) {

	    int number_of_additions=0;

	    
	    
	    for(int i=0;i<no_attempts;i++) {
		long row_or_col = rand()%1000000;
		if((double)row_or_col/1000000<=ratio_of_col) {
		    if(shuffle_columns()) {
			number_of_additions++;
		    }
		} else {
		    if(shuffle_rows()) {
			number_of_additions++;
		    }
		    
		}
		
	    }
	    return number_of_additions;
	}
		
	// Precondition: the Graded_matrix represents a minimal presentation!
	// Precondition: the number of parameters is 2
	// Side effect: This sorts the rows and columns in lex order!
	bool is_interval() {
	    assert(number_of_parameters==2);
	    to_lex_order(*this,true,false);
	    for(int i=0;i<row_grades.size()-1;i++) {
		assert(row_grades[i].at[0]<=row_grades[i+1].at[0]);
		if(row_grades[i].at[0]==row_grades[i+1].at[0]) {
		    return false;
		}
		if(row_grades[i].at[1]<=row_grades[i+1].at[1]) {
		    return false;
		}
	    }
	    // Check that there are relations on the joins
	    int rel_idx=0;
	    for(int i=0;i<row_grades.size()-1;i++) {
		double join_x=row_grades[i+1].at[0];
		double join_y=row_grades[i].at[1];

		while(grades[rel_idx].at[0]<join_x) {
		    rel_idx++;
		}
		bool relation_found=false;
		while(grades[rel_idx].at[0]==join_x) {
		    if(grades[rel_idx].at[1]!=join_y) {
			rel_idx++;
			continue;
		    }
		    std::vector<long> bd;
		    this->get_col(rel_idx,bd);
		    if(bd.size()!=2) {
			rel_idx++;
			continue;
		    }
		    if(bd[0]==i || bd[1]==i+1) {
			relation_found=true;
			break;
		    }
		}
		if(!relation_found) {
		    return false;
		}		
	    }
	    
	    return true;
	}
	
    }; // of class Graded_matrix


    template<typename GradedMatrix>
    void check_boundaries(GradedMatrix& M, std::string msg) {
	for(index i=0;i<M.get_num_cols();i++) {
	    std::vector<index> col;
	    M.get_col(i,col);
	    for(index j=1;j<col.size();j++) {
		if(col[j-1]>=col[j]) {
		    std::cout << "Bad boundary " << col[j-1] << " " << col[j] << " at column " << i << "(" << msg << " matrix)" << std::endl;
		    std::exit(1);
		}
	    }
	}
    }


    
    template<typename GradedMatrix>
    void check_index_sanity(GradedMatrix& M) {
	int num_rows = M.num_rows;
	
	for(index i=0;i<M.get_num_cols();i++) {
	    std::vector<index> col;
	    M.get_col(i,col);
	    for(index j : col) {
		if(j<0 || j>=num_rows) {
		    std::cout << "Bad index in column " << i << ": Row index " << j << " appears but the matrix has only " << num_rows << " rows" << std::endl;
		    std::exit(1);
		}
	    }
	}
    }

	
    template<typename GradedMatrix>
    void check_grade_sanity(GradedMatrix& M) {
      
	typedef typename GradedMatrix::Grade Grade;
      
	for(index i=0;i<M.get_num_cols();i++) {
	    Grade& col_gr = M.grades[i];
	    std::vector<index> col;
	    M.get_col(i,col);
	    for(index j=0;j<col.size();j++) {
		Grade& row_gr = M.row_grades[col[j]];
		for(int p=0;p<M.number_of_parameters;p++) {
		    if(row_gr.at[p]>col_gr.at[p]) {
			std::cout << "Bad grading at: " << col[j] << "( at " << p << " " << row_gr.at[p] << ") " << i << "( " << col_gr.at[p] << ")" << std::endl;
			std::exit(1);
		    }
		    assert(row_gr.at[p]<=col_gr.at[p]);
		}
	    }
	}
    }

    template<typename GradedMatrix>
    void check_chain_complex_sanity(std::vector<GradedMatrix>& matrices) {

	std::cout << "Checking sanity" << std::endl;
	
	for(int i=0;i<matrices.size();i++) {
	    //std::cout << "Basic checks " << i << std::endl;
	    check_grade_sanity(matrices[i]);
	    check_index_sanity(matrices[i]);
	}
	for(int i=0;i<matrices.size()-1;i++) {
	    //std::cout << "Bd of bd checks " << i << std::endl;
	    //std::cout << "Size: " << matrices[i].get_num_cols() << std::endl;
	    //std::cout << "Next matrix has size " << matrices[i+1].get_num_cols() << std::endl;
	    for(int j=0;j<matrices[i].get_num_cols();j++) {
		//std::cout << "j=" << j << std::endl;
		std::vector<index> bd_of_bd;
		std::vector<index> bd;
		matrices[i].get_col(j,bd);
		for(index k : bd) {
		    //std::cout << "Check entry " << k << std::endl;
		    std::vector<index> bd_col;
		    matrices[i+1].get_col(k,bd_col);
		    std::copy(bd_col.begin(),bd_col.end(),std::back_inserter(bd_of_bd));
		}
		std::sort(bd_of_bd.begin(),bd_of_bd.end());
		if(bd_of_bd.size()%2==1) {
		    std::cerr << "Entry has an odd number of codim-2-faces (counted with multiplicity)" << std::endl;
		    std::exit(1);
		}
		for(int k=0;k<bd_of_bd.size();k+=2) {
		    if(bd_of_bd[k]!=bd_of_bd[k+1]) {
			std::cerr << "codim-2-faces do not cancel out" << std::endl;
			std::exit(1);
		    }
		}
	    }
	}
	for(int j=0;j<matrices.back().get_num_cols();j++) {
	    std::vector<index> bd;
	    matrices.back().get_col(j,bd);
	    if(bd.size()!=0) {
		std::cerr << "Column in last matrix has nonz-zero boundary" << std::endl;
		std::exit(1);
	    }
	}
	
    }
    
    template<typename InputIterator>
    void copy_grades_to_rows(InputIterator begin,
			     InputIterator end) {

	typedef typename InputIterator::value_type GradedMatrix;
	auto it1=begin, it2=begin;
	it2++;
	while(it2!=end) {
	    GradedMatrix& M1=*it1;
	    GradedMatrix& M2=*it2;
	    assert(M1.num_rows==M2.get_num_cols());
	    M1.row_grades.clear();
	    for(index j=0;j<M1.num_rows;j++) {
		M1.row_grades.push_back(M2.grades[j]);
	    }
	    //std::cout << "Copied " << M1.num_rows << " grades" << std::endl;
	    it1++;
	    it2++;
	}
    
    }



    template<typename InputIterator>
    void assign_grade_indices(InputIterator begin,
			      InputIterator end) {
    
	typedef typename InputIterator::value_type GradedMatrix;
	typedef typename GradedMatrix::Coordinate Coordinate;

	//std::cout << "Assgin grade indices with " << n1 << ", " << n2 << " columns" << std::endl;
    
	std::unordered_map<Coordinate,index> val_to_index_x, val_to_index_y;
    
	std::vector<Coordinate> x_vals,y_vals;

	for(auto it=begin;it!=end;it++) {
	    GradedMatrix& matrix=*it;

	    int n = matrix.get_num_cols();
	    for(int i=0;i<n;i++) {
		x_vals.push_back(matrix.grades[i].at[0]);
		y_vals.push_back(matrix.grades[i].at[1]);
	    }
      
	}
    
	std::sort(x_vals.begin(),x_vals.end());
	auto last_x = std::unique(x_vals.begin(),x_vals.end());
	x_vals.erase(last_x,x_vals.end());
	std::sort(y_vals.begin(),y_vals.end());
	auto last_y = std::unique(y_vals.begin(),y_vals.end());
	y_vals.erase(last_y,y_vals.end());
    
	if(verbose) std::cout << "Found " << x_vals.size() << " different x-values and " << y_vals.size() << " different y-values" << std::endl;
	for(auto it=begin;it!=end;it++) {
	    GradedMatrix& matrix=*it;
	    matrix.x_vals.clear();
	    std::copy(x_vals.begin(),x_vals.end(),std::back_inserter(matrix.x_vals));
	    matrix.y_vals.clear();
	    std::copy(y_vals.begin(),y_vals.end(),std::back_inserter(matrix.y_vals));
	    matrix.num_grades_x = x_vals.size();
	    matrix.num_grades_y = y_vals.size();
	    //std::cout << "num grade x: " <<  matrix.num_grades_x << std::endl;
	    //std::cout << "num grade y: " <<  matrix.num_grades_y << std::endl;
	}
    
	for(index i=0;i<x_vals.size();i++) {
	    val_to_index_x[x_vals[i]]=i;
	}
	for(index i=0;i<y_vals.size();i++) {
	    val_to_index_y[y_vals[i]]=i;
	}
	long total_count=0;
	for(auto it=begin;it!=end;it++) {
	    GradedMatrix& matrix = *it;
	    int n = matrix.get_num_cols();
	    //std::cout << "Handling " << n << " columns " << std::endl;
	    total_count+=n;
	    for(int i=0;i<n;i++) {
		matrix.grades[i].index_at[0]=val_to_index_x[matrix.grades[i].at[0]];
		assert(matrix.grades[i].at[0] < matrix.num_grades_x);
		matrix.grades[i].index_at[1]=val_to_index_y[matrix.grades[i].at[1]];
		assert(matrix.grades[i].at[1] < matrix.num_grades_y);
	    }
	}
	//std::cout << "n1=" << n1 << std::endl;
	//std::cout << "n2=" << n2 << std::endl;
	if(verbose) std::cout << "N is " << total_count << std::endl;
	copy_grades_to_rows(begin,end);
	for(auto it=begin;it!=end;it++) {
	    it->grade_indices_assigned=true;
	}
    }


  
    template<typename GradedMatrix>
    void assign_grade_indices_of_pair(GradedMatrix& M1, GradedMatrix& M2) {
    
	typedef typename GradedMatrix::Coordinate Coordinate;

	int n1=M1.get_num_cols();
	int n2=M2.get_num_cols();

	//std::cout << "Assgin grade indices with " << n1 << ", " << n2 << " columns" << std::endl;
	if(n1==0 && n2==0) {
	    return;
	}
    
	std::unordered_map<Coordinate,index> val_to_index_x, val_to_index_y;
    
	std::vector<Coordinate> x_vals,y_vals;
    
	for(int i=0;i<n1;i++) {
	    x_vals.push_back(M1.grades[i].at[0]);
	    y_vals.push_back(M1.grades[i].at[1]);
	}
	for(int i=0;i<n2;i++) {
	    x_vals.push_back(M2.grades[i].at[0]);
	    y_vals.push_back(M2.grades[i].at[1]);
	}
    
	std::sort(x_vals.begin(),x_vals.end());
	auto last_x = std::unique(x_vals.begin(),x_vals.end());
	x_vals.erase(last_x,x_vals.end());
	std::sort(y_vals.begin(),y_vals.end());
	auto last_y = std::unique(y_vals.begin(),y_vals.end());
	y_vals.erase(last_y,y_vals.end());
    
	if(verbose) std::cout << "Found " << x_vals.size() << " different x-values and " << y_vals.size() << " different y-values" << std::endl;
	M1.x_vals=x_vals;
	M1.y_vals=y_vals;
	M1.num_grades_x = x_vals.size();
	M1.num_grades_y = y_vals.size();
	M2.x_vals=x_vals;
	M2.y_vals=y_vals;
	M2.num_grades_x = x_vals.size();
	M2.num_grades_y = y_vals.size();
    
	for(index i=0;i<x_vals.size();i++) {
	    val_to_index_x[x_vals[i]]=i;
	}
	for(index i=0;i<y_vals.size();i++) {
	    val_to_index_y[y_vals[i]]=i;
	}
    
	for(int i=0;i<n1;i++) {
	    M1.grades[i].index_at.resize(2);
	    M1.grades[i].index_at[0]=val_to_index_x[M1.grades[i].at[0]];
	    M1.grades[i].index_at[1]=val_to_index_y[M1.grades[i].at[1]];
	}
	for(int i=0;i<n2;i++) {
	    M2.grades[i].index_at.resize(2);
	    M2.grades[i].index_at[0]=val_to_index_x[M2.grades[i].at[0]];
	    M2.grades[i].index_at[1]=val_to_index_y[M2.grades[i].at[1]];
	}
	M1.row_grades.clear();
	for(int i=0;i<n2;i++) {
	    M1.row_grades.push_back(M2.grades[i]);
	}
	M1.grade_indices_assigned=true;
	M2.grade_indices_assigned=true;
    
	//std::cout << "n1=" << n1 << std::endl;
	//std::cout << "n2=" << n2 << std::endl;
	if(verbose) std::cout << "N is " << n1+n2 << std::endl;
    }

  
    template<typename GradedMatrix>
    void assign_grade_indices(GradedMatrix& M1) {
	GradedMatrix M2;
	M2.set_dimensions(0,M1.num_rows);
	M2.grades=M1.row_grades;
	assign_grade_indices_of_pair(M1,M2);
    }
  
    template<typename GradedMatrix,typename OutStream>
    void print_in_scc_format(std::vector<GradedMatrix>& matrices,
			     OutStream& outstr,
			     bool extend_by_zero_matrix=false,
			     bool bring_to_lex_order=false) {
    
	outstr << "scc2020\n";
	int no_parameters = matrices.size() >0 ? matrices[0].number_of_parameters : 0;
	outstr << no_parameters << "\n";
	for(index d=0;d<matrices.size();d++) {
	    outstr << matrices[d].get_num_cols() << " ";
	}
	outstr << matrices[matrices.size()-1].num_rows << " ";
	if(extend_by_zero_matrix) {
	    outstr << "0";
	}
	outstr << "\n";

	for(index d=0;d<matrices.size()-1;d++) {
	    GradedMatrix& M = matrices[d];
	    if(bring_to_lex_order) {
		to_lex_order(M,true,false);
	    }
	    M.print_in_rivet_format(outstr,false,false); // false for "print no header", false for "print no rows"
	}
	if(bring_to_lex_order) {
	    to_lex_order(matrices.back(),false,false);
	}
	matrices.back().print_in_rivet_format(outstr,false,extend_by_zero_matrix);
      
    }

    // Requires that the vector is sorted such that equal entries appear consecutively (e.g. lex- or colex sorted)
    template<typename Grade>
    index max_number_of_entries_with_same_grade(std::vector<Grade>& grades) {
    
	if(grades.size()==0) {
	    return 0;
	}
    
	int max_k=1;
    
	int curr_k=1;

	Grade& last_bigrade = grades[0];

	for(int i=1;i<grades.size();i++) {
	    if(grades[i]==last_bigrade) {
		curr_k++;
	    } else {
		last_bigrade = grades[i];
		if(curr_k>max_k) {
		    max_k=curr_k;
		}
		curr_k=1;
	    }
	}
	if(curr_k>max_k) {
	    max_k=curr_k;
	}
	return max_k;
    }

    template<typename GradedMatrix>
    index max_number_of_columns_with_same_grade(GradedMatrix& M) {

	return max_number_of_entries_with_same_grade(M.grades);

    }

    template<typename GradedMatrix>
    index max_number_of_rows_with_same_grade(GradedMatrix& M) {

	return max_number_of_entries_with_same_grade(M.row_grades);

    }


  
}//of namespace mpp_utils
