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

#ifndef MPFREE_TIMERS
#define MPFREE_TIMERS 0
#endif

#include <limits>

#if MPFREE_TIMERS
#include <mpfree/boost_timers.h>
#endif

#include<scc/Scc.h>
#include<mpp_utils/create_graded_matrices_from_scc2020.h>

#include<mpp_utils/create_matrix_from_firep.h>
#include<mpp_utils/Coordinate_traits_with_map.h>
#include<mpp_utils/Graded_matrix_handle.h>
#include<mpp_utils/sorting_utility.h>

#include<mpfree/global.h>

#include<mpfree/Graded_matrix.h>
#include<mpfree/mpfree_subroutines.h>


namespace mpfree {
  
  template<
    typename GradedMatrix
    >
    void compute_minimal_presentation(GradedMatrix& input_GM1,
				      GradedMatrix& input_GM2,
				      GradedMatrix& min_rep,
				      bool use_chunk=true,
				      bool use_clearing=false
				      ) {
    
    typedef GradedMatrix Graded_matrix;
    
    typedef typename mpfree::Extend_matrix<GradedMatrix>::Type Graded_matrix_extended;

    if(! mpp_utils::is_colex_sorted(input_GM1)) {
	mpp_utils::to_colex_order(input_GM1,true,false);
	input_GM1.grade_indices_assigned=false;
    }
    if(! mpp_utils::is_colex_sorted_columns(input_GM2)) {
	mpp_utils::to_colex_order(input_GM2,false,false);
	input_GM2.grade_indices_assigned=false;
    }
    
    if(!input_GM1.grade_indices_assigned || !input_GM2.grade_indices_assigned) {
	//std::cout << "here" << std::endl;
      mpp_utils::assign_grade_indices_of_pair(input_GM1,input_GM2);
    }

    

    
    Graded_matrix_extended GM1(&input_GM1);
    Graded_matrix_extended GM2(&input_GM2);



    GM1.assign_slave_matrix();
    GM1.assign_pivots();
    GM2.assign_slave_matrix();
    GM2.assign_pivots();
    GM1.pq_row.resize(GM1.num_grades_y);
    GM2.pq_row.resize(GM2.num_grades_y);
    
    //std::cout << "Memory after reading input: " << mem_info() << std::endl;
    
    if(use_chunk) {
      
#if MPFREE_TIMERS
      chunk_timer.resume();
#endif
      
      if(mpfree::verbose) std::cout << "Chunk preprocessing..." << std::flush;
      //test_timer1.start();
      chunk_preprocessing(GM1,GM2);
      //test_timer1.stop();
      if(mpfree::verbose) std::cout << "done" << std::endl;
      
#if MPFREE_TIMERS
      chunk_timer.stop();
#endif
      
      //std::cout << "Memory after chunk: " << mem_info() << std::endl;
      
    }
    Graded_matrix MG_base,Ker_base;
    Graded_matrix_extended MG(&MG_base),Ker(&Ker_base);

   
#if MPFREE_TIMERS
    mingens_timer.resume();
#endif
    if(mpfree::verbose) std::cout << "Min Gens..." << std::flush;
    GM1.grid_scheduler=mpfree::Grid_scheduler(GM1);
    min_gens(GM1,MG,use_clearing);
    
    if(mpfree::verbose) std::cout << "done, size is " << MG.num_rows << "x" << MG.get_num_cols() << std::endl;
    input_GM1=Graded_matrix();
#if MPFREE_TIMERS
    mingens_timer.stop();
#endif
    
    //std::cout << "Memory after Min gens: " << mem_info() << std::endl;
    
#if MPFREE_TIMERS
    kerbasis_timer.resume();
#endif
    
    if(mpfree::verbose) std::cout << "Ker basis..." << std::flush;
    
    

    GM2.grid_scheduler=mpfree::Grid_scheduler(GM2);

    ker_basis(GM2,Ker,MG,use_clearing);
    if(mpfree::verbose) std::cout << "done, size is " << Ker.num_rows << "x" << Ker.get_num_cols() << std::endl;
    input_GM2=Graded_matrix(); 
#if MPFREE_TIMERS
    kerbasis_timer.stop();
#endif
    
    //std::cout << "Memory after ker basis: " << mem_info() << std::endl;
    
#if MPFREE_TIMERS
    reparam_timer.resume();
#endif
    Graded_matrix semi_min_rep_base;
    Graded_matrix_extended semi_min_rep(&semi_min_rep_base);
    if(mpfree::verbose) std::cout << "Reparameterize..." << std::flush;
    reparameterize(MG,Ker,semi_min_rep);
    if(mpfree::verbose) std::cout << "done" << std::endl;
    if(mpfree::verbose) std::cout << "Resulting semi-minimal presentation has " << semi_min_rep.get_num_cols() << " columns and " << semi_min_rep.num_rows << " rows" << std::endl;
    MG_base=Graded_matrix();
    Ker_base=Graded_matrix();
    
#if MPFREE_TIMERS
    reparam_timer.stop();
#endif
    
    //std::cout << "Memory after reparam: " << mem_info() << std::endl;
    
    
#if MPFREE_TIMERS
    minimize_timer.resume();
#endif
    if(mpfree::verbose) std::cout << "Minimize..." << std::flush;
    minimize(semi_min_rep,min_rep);
    if(mpfree::verbose) std::cout << "done" << std::endl;
    if(mpfree::verbose) std::cout << "Resulting minimal presentation has " << min_rep.get_num_cols() << " columns and " << min_rep.num_rows << " rows" << std::endl;
    semi_min_rep_base=Graded_matrix(); 
#if MPFREE_TIMERS
    minimize_timer.stop();
#endif
    
    //std::cout << "Memory after minimize: " << mem_info() << std::endl;
    

    
    
    
    //std::cout << "Memory at termination: " << mem_info() << std::endl;
  }

  template<typename GradedMatrix>
    void compute_minimal_presentation(GradedMatrix& GM1,
				      GradedMatrix& GM2,
				      const std::string& outfile,
				      bool output_in_scc_format=true,
				      bool use_chunk=true,
				      bool use_clearing=false
				      ) {
    typedef GradedMatrix Graded_matrix;
    Graded_matrix min_rep;
    compute_minimal_presentation(GM1,GM2,min_rep,use_chunk,use_clearing);
    if(outfile!="") {
#if MPFREE_TIMERS
      io_timer.resume();
#endif
      
      if(mpfree::verbose) std::cout << "Writing to file \"" << outfile << "\"..." << std::flush;
      std::ofstream ofstr((char*)outfile.c_str());
#ifdef HIGH_PRECISION_OUTPUT
      ofstr.precision(std::numeric_limits<double>::max_digits10);
#endif
      if(output_in_scc_format) {
	std::vector<Graded_matrix> matrices;
	matrices.push_back(min_rep);
	mpp_utils::print_in_scc_format(matrices,ofstr,true); // true for printing the rows of last entry
      } else {
	min_rep.print_in_rivet_format(ofstr);
      }
      ofstr.close();
      if(mpfree::verbose) std::cout << "done" << std::endl;
#if MPFREE_TIMERS
      io_timer.stop();
#endif
      
    }
  }



  template< typename GrMat > 
    void get_matrices_from_file(const std::string& infile,
				std::vector<GrMat>& matrices,
				int start=1,
				int end=2) {
  
    // We could just call the stream version but in the firep case
    // reading a file might be more efficient than reading an actual
    // stream. As long as this is not tested, keep both versions
    // separate
  
    std::string magic_word=scc::get_magic_word(infile);
    
    if(magic_word=="scc2020") {
      
      std::cout << "scc2020 file" << std::endl;
      
      std::ifstream ifstr(infile);
      typedef typename GrMat::Coordinate_traits Grade_cast;
      scc::Scc<Grade_cast> parser(ifstr,&Grade_cast::get_instance());
      if(start==-1) {
	mpp_utils::create_graded_matrices_from_scc2020(parser,matrices);
      } else {
	mpp_utils::create_graded_matrices_from_scc2020(parser,start,end,matrices);
      }
      
      parser.clear();
    } else if(magic_word=="firep") {
      //std::cout << "firep file" << std::endl;
      matrices.resize(2);
      mpp_utils::create_matrix_from_firep(infile,matrices[0],matrices[1]);
    } else {
      std::cerr << "File type not recognized (firep or scc2020 expected)" << std::endl;
      std::exit(1);
    }
    
  }

  template< typename GrMat > 
    void get_matrices_from_stream(std::istream& instr,
				  std::vector<GrMat>& matrices,
				  int start=1,
				  int end=2) {
    
    std::string magic_word=scc::get_magic_word(instr);
    
    if(magic_word=="scc2020") {
      
      std::cout << "scc2020 file" << std::endl;
      
      typedef typename GrMat::Coordinate_traits Grade_cast;
      scc::Scc<Grade_cast> parser(instr,&Grade_cast::get_instance());
      if(start==-1) {
	mpp_utils::create_graded_matrices_from_scc2020(parser,matrices);
      } else {
	mpp_utils::create_graded_matrices_from_scc2020(parser,start,end,matrices);
      }
      
      parser.clear();
    } else if(magic_word=="firep") {
      //std::cout << "firep file" << std::endl;
      matrices.resize(2);
      mpp_utils::create_matrix_from_firep(instr,matrices[0],matrices[1]);
    } else {
      std::cerr << "File type not recognized (firep or scc2020 expected)" << std::endl;
      std::exit(1);
    }
    
  }

  

  template<typename GradedMatrix>
    void compute_minimal_presentation(const std::string& infile,
				      const std::string& outfile,
				      bool output_in_scc_format=true,
				      int dim=1,
				      bool use_chunk=true,
				      bool use_clearing=false) {
    
    typedef GradedMatrix Graded_matrix;
    
#if _OPENMP
    if(mpfree::verbose) std::cout << "Execution parralized, max Number of threads: " << omp_get_max_threads() << std::endl;
#endif
    
    //std::cout << "Memory initially: " << mem_info() << std::endl;
    
#if MPFREE_TIMERS
    io_timer.resume();
#endif
    
    std::vector<Graded_matrix> matrices;
    get_matrices_from_file<Graded_matrix>(infile,matrices,dim,dim+1);
    


#if MPFREE_TIMERS
    io_timer.stop();
#endif


    compute_minimal_presentation(matrices[0],
				 matrices[1],
				 outfile,
				 output_in_scc_format,
				 use_chunk,
				 use_clearing);
  }
  
  template<typename GradedMatrix>
    void compute_free_resolution(const std::string& infile,
				 const std::string& outfile,
				 int dim=0,
				 bool use_chunk=true,
				 bool use_clearing=false) {

    typedef GradedMatrix Graded_matrix;
#if MPFREE_TIMERS
    io_timer.resume();
#endif
    
    std::vector<Graded_matrix> matrices;
    get_matrices_from_file<Graded_matrix>(infile,matrices,dim,dim+1);
    

#if MPFREE_TIMERS
    io_timer.stop();
#endif

    Graded_matrix min_pres;
    compute_minimal_presentation(matrices[0],
				 matrices[1],
				 min_pres,
				 use_chunk,
				 use_clearing);

    mpp_utils::to_colex_order(min_pres);

    std::vector<Graded_matrix> results;
    results.resize(2);
    results[1]=min_pres;

#if MPFREE_TIMERS
    syzygy_timer.resume();
#endif
    Graded_matrix syzygy_matrix,dummy;
    typedef typename mpfree::Extend_matrix<Graded_matrix>::Type Graded_matrix_extended;

    Graded_matrix_extended min_pres_ext(&min_pres);
    Graded_matrix_extended dummy_ext(&dummy);
    Graded_matrix_extended syzygy_matrix_ext(&syzygy_matrix);

    min_pres_ext.grid_scheduler = mpfree::Grid_scheduler(min_pres);
    min_pres_ext.pq_row.resize(min_pres.num_grades_y);
    min_pres_ext.assign_pivots();
    min_pres_ext.assign_slave_matrix();
    //std::cout << "SIZE of pq_row: " << min_pres.pq_row.size() << std::endl;
    if(mpfree::verbose) std::cout << "Computing syzygy.." << std::flush;
    ker_basis(min_pres_ext,syzygy_matrix_ext,dummy_ext,false);
    if(mpfree::verbose) std::cout << "done, found " << syzygy_matrix.get_num_cols() << " syzygies" << std::endl;
#if MPFREE_TIMERS
    syzygy_timer.stop();
#endif
    results[0]=syzygy_matrix;
#if MPFREE_TIMERS
    io_timer.resume();
#endif
    std::ofstream ofstr((char*)outfile.c_str());
    mpp_utils::print_in_scc_format(results,ofstr,true);
    ofstr.close();
#if MPFREE_TIMERS
    io_timer.stop();
#endif
    

  }

  
} //of namespace mpfree
