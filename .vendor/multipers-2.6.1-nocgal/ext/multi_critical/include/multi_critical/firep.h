/* Copyright 2025 TU Graz
   Author: Michael Kerber
   
   This file is part of multi_critical
   
   multi_critical is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   multi_critical is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with multi_critical.  If not, see <https://www.gnu.org/licenses/>.*/

#pragma once

#include<algorithm>
#include<limits>

#include<mpp_utils/Pre_column_struct.h>
#include<mpp_utils/create_graded_matrices_from_pre_column_struct.h>

#include<multi_critical/basic.h>
#include<multi_critical/Grade.h>

namespace multi_critical {

    
    
    template<typename ParserType, typename GradedMatrix>
	void firep_via_csv(ParserType& parser,
			   int dim,
			   GradedMatrix& GM1,
			   GradedMatrix& GM2,
			   bool interpret_dim_as_hom_dim=true) {

	// dim means the homology dimension which does not
	// correspond to the ordering in the parser

	int d = parser.number_of_parameters();

	if(d!=2) {
	    std::cerr << "This code expects a 2-parameter input" << std::endl;
	    return;
	}

	int ell = parser.number_of_levels();

	int k = interpret_dim_as_hom_dim ? ell-dim : dim;

	typedef typename GradedMatrix::Grade Matrix_Grade;
	typedef mpp_utils::Pre_column_struct<Matrix_Grade> Pre_column;

	std::vector<std::vector<Pre_column>> pre_columns;
	pre_columns.resize(2);

	index gen_idx=0;
	index rel_idx=0;
	
	std::vector<std::vector<long> > copies_of_k_critical_generators;

	if(multi_critical::very_verbose) {
	    std::cout << "INFO: #Generators k-critical: ";
	    if(k>1) {
		std::cout << parser.number_of_generators(k-1) << " ";
	    } else {
		std::cout << "0 ";
	    }
	    std::cout << parser.number_of_generators(k) << " ";
	    if(k<parser.number_of_levels()) {
		std::cout << parser.number_of_generators(k+1) << " ";
	    } else {
		std::cout << "0 ";
	    }
	    std::cout << std::endl;
	}

	if(multi_critical::verbose) std::cout << "Compute (k+1)-k-boundary matrix" << std::endl;
	
	while(parser.has_next_column(k)) {
	    std::vector<double> raw_grades;
	    std::vector<std::pair<long,int>> boundary_and_coeffs;
	    parser.next_column(k,
			       std::back_inserter(raw_grades),
			       std::back_inserter(boundary_and_coeffs));
	    // Ignore coefficients
	    std::vector<long> boundary;
	    for(auto& x : boundary_and_coeffs) {
		boundary.push_back(x.first);
	    }
	    std::vector<Grade> grades;
	    get_incomparable_grades_in_lex_order(raw_grades,grades);
	    //std::cout << "Generator is " << grades.size() << "-critical" << std::endl;
	    // Create the generators
	    std::vector<long> copies_of_this_idx;
	    for(int j=0;j<grades.size();j++) {
		Grade& gr=grades[j];
		Matrix_Grade mat_gr(gr.x,gr.y);
		Pre_column new_precolumn(gen_idx,mat_gr,boundary);
		copies_of_this_idx.push_back(gen_idx);
		pre_columns[1].push_back(new_precolumn);
		if(j>0) {
		    // Create relations at join
		    assert(!are_comparable(grades[j-1],grades[j]));
		    Grade curr_join=join(grades[j-1],grades[j]);
		    Matrix_Grade curr_mat_join(curr_join.x,curr_join.y);
		    std::vector<long> curr_boundary;
		    curr_boundary.push_back(gen_idx-1);
		    curr_boundary.push_back(gen_idx);
		    Pre_column new_rel_precolumn(rel_idx,curr_mat_join,curr_boundary);
		    pre_columns[0].push_back(new_rel_precolumn);
		    rel_idx++;
		}
		gen_idx++;
	    }
	    copies_of_k_critical_generators.push_back(copies_of_this_idx);
	}
	// Now put copies of the relations: The only problem
	// is that the boundary needs to be adapted

	if(multi_critical::verbose) std::cout << "Compute (k+1)-k-boundary matrix" << std::endl;

	if(k!=1) { // If k==0, no relations
	    while(parser.has_next_column(k-1)) {
		std::vector<double> raw_grades;
		std::vector<std::pair<long,int>> boundary_and_coeffs;
		parser.next_column(k-1,
				   std::back_inserter(raw_grades),
				   std::back_inserter(boundary_and_coeffs));
		// Ignore coefficients
		std::vector<long> boundary;
		for(auto& x : boundary_and_coeffs) {
		    boundary.push_back(x.first);
		}
		std::vector<Grade> grades;
		get_incomparable_grades_in_lex_order(raw_grades,grades);

		//std::cout << "Relation is " << grades.size() << "-critical" << std::endl;
		for(int j=0;j<grades.size();j++) {
		    Grade& gr=grades[j];
		    //std::cout << "Grade " << gr.x << " " << gr.y << std::endl;
		    std::vector<long> bd_1_critical;
		    for(long b : boundary) {
			//std::cout << "b=" << b << std::endl;
			std::vector<long>& copies=copies_of_k_critical_generators[b];

			bool el_found=false;
			for(long c : copies) {
			    Grade c_gr(pre_columns[1][c].grade.at[0],
				       pre_columns[1][c].grade.at[1]);
			    //std::cout << "Compare grade: " << c_gr.x << " " << c_gr.y << std::endl;
			    if(c_gr<=gr) {
				bd_1_critical.push_back(c);
				el_found=true;
				break;
			    }
			}
			if(! el_found) {
			    std::cerr << "No element found!!" << std::endl;
			    std::cerr << "Boundary:";
			    for(auto i: boundary)  {
				std::cout << i << " ";
			    }
			    std::cout << std::endl;
			    std::cout << "Missed b=" << b << std::endl;
			}
		    }
		    Matrix_Grade mat_gr(gr.x,gr.y);
		    pre_columns[0].push_back(Pre_column(rel_idx,
							mat_gr,
							bd_1_critical));
		    rel_idx++;
		}
	    }
	}

	if(multi_critical::verbose) std::cout << "#Generators: " << pre_columns[1].size() << std::endl;
	if(multi_critical::verbose) std::cout << "#Relations:  " << pre_columns[0].size() << std::endl;

	long number_of_rows_in_last_matrix=(k<parser.number_of_levels()) ? parser.number_of_generators(k+1) : 0;

	std::vector<GradedMatrix> graded_matrices;

	mpp_utils::create_graded_matrices_from_pre_column_struct(pre_columns,
								graded_matrices,
								 number_of_rows_in_last_matrix,
								 false,
								 false);
	GM1=graded_matrices[0];
	GM2=graded_matrices[1];

	if(multi_critical::verbose) std::cout << "#Generators (1-crit): " << gen_idx << std::endl;
	if(multi_critical::verbose) std::cout << "#Relations (1-crit): " << rel_idx << std::endl;

	/*
	{
	    std::ofstream ofstr("debug.txt");
	    ofstr.precision(std::numeric_limits<double>::max_digits10);
	    print_in_scc_format(graded_matrices,ofstr,false,false);
	    ofstr.close();
	}
	{
	    std::ofstream ofstr("debug_ordered.txt");
	    ofstr.precision(std::numeric_limits<double>::max_digits10);
	    print_in_scc_format(graded_matrices,ofstr,false,true);
	    ofstr.close();
	}
	*/
	return;
    }


    
}
