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
#include<scc/Scc.h>

#include<mpp_utils/create_graded_matrices_from_scc2020.h>
#include<multi_chunk/Graded_matrix.h>

#ifndef MULTI_CHUNK_TIMERS
#define MULTI_CHUNK_TIMERS 0
#endif

#if MULTI_CHUNK_TIMERS
#include <multi_chunk/boost_timers.h>
#endif


namespace multi_chunk {

  template<typename GrMat> void compress(std::vector<GrMat>& input_matrices) {


    index max_d = input_matrices.size()-2; // the last matrix has no row grades, do not use chunk here

    if(multi_chunk::verbose) std::cout << "Initialize.." << std::endl;
    
    long N=0;

    typedef typename Extend_matrix<GrMat>::Type GrMatExt;

    std::vector<GrMatExt> matrices;

    for(index i=0;i<input_matrices.size();i++) {
      matrices.push_back(GrMatExt(&(input_matrices[i])));
      matrices[i].init();
      N+=matrices[i].get_num_cols();
      if(multi_chunk::verbose) std::cout << "Num rows " << i << ": " << matrices[i].num_rows << std::endl;
      //matrices[i].print(false,false);
    }
    

    
#if MULTI_CHUNK_TIMERS
    local_reduction_timer.start();
#endif
    if(multi_chunk::verbose) std::cout << "Local reductions..." << std::endl;
    for(index d=0;d<=max_d;d++) {
      GrMatExt& M=matrices[d];
      if(multi_chunk::verbose) std::cout << "d=" << d << ", num cols: " << M.get_num_cols() << ", num rows: " << M.row_grades.size() << std::endl;
      index gl_index=0;
      for(index i=0;i<M.get_num_cols();i++) {
	/*
	  if(d>=0) {
	  std::vector<index> col;
	  M.get_col(i,col);
	  std::cout << "Col size: " << col.size() << std::endl;
	  if(col.size()>0) {
	  for(index j : col) {
	  std::cout << j << " ";
	  }
	  std::cout << std::endl;
	  }
	  }
	*/
	if(M.is_empty(i)) {
	  if(M.status[i]==0) {
	    M.status[i]=2;
	    M.global_index[i]=gl_index++;
	  }
	  continue;
	}
	index k = M.get_max_index(i);
	index l = M.pivots[k];
	while(M.is_local(i) && l!=-1) {
	  M.add_to(l,i);
	  k = M.get_max_index(i);
	  l = M.pivots[k];
	}
	if(M.is_local(i)) {
	  M.pivots[k]=i;
	  M.status[i]=-1;
	  matrices[d+1].status[k]=1;
	  matrices[d+1].clear(k); // clearing
	  //std::cout << "Local pair: " << d << " - " << i << " and " << d+1 << " - " << k << std::endl;
	} else {
	  if(M.status[i]==0) {
	    M.status[i]=2;
	    M.global_index[i]=gl_index++;
	  }
	}
      }
      if(multi_chunk::verbose) std::cout << "# global indices " << gl_index << std::endl;
      if(multi_chunk::verbose) std::cout << "Num entries after local reduce: " << M.get_num_entries() << std::endl;
    }
    // Manually set global indices for the last matrix
    {
      GrMatExt& M = matrices[matrices.size()-1];
      index gl_index=0;
      for(index i=0;i<M.get_num_cols();i++) {
	if(M.status[i]==0) {
	  M.status[i]=2;
	}
	if(M.status[i]==2) {
	  M.global_index[i]=gl_index++;
	}
      }
    }
    
#if MULTI_CHUNK_TIMERS
    local_reduction_timer.stop();
#endif

#if MULTI_CHUNK_TIMERS
    sparsification_timer.start();
#endif

    if(multi_chunk::verbose) std::cout << "Compression..." << std::endl;
    for(index d=max_d;d>=0;d--) {
      GrMatExt& M=matrices[d];
#pragma omp parallel for schedule(guided,1)
      for(index i=0;i<M.get_num_cols();i++) {
	assert(M.status[i]!=0);
	if(M.status[i]!=2) {
	  continue;
	}
	std::vector<index> col;
	while(!M.is_empty(i)) {
	  index p = M.get_max_index(i);
	  index j = M.pivots[p];
	  if(j!=-1) {
	    assert(M.status[j]==-1);
	    M.add_to(j,i);
	  } else {
	    index p_status=matrices[d+1].status[p];
	    if(p_status==0 || p_status==1) {
	      std::cout << "Bad status " << p_status << " at d=" << d << ", i=" << i <<", p=" << p << std::endl;
	    }
	    assert(p_status==-1 || p_status==2);
	    if(p_status==2) {
	      //std::cout << "Info: " << p << " " << (matrices[d+1].global_index[p]) << std::endl;
	      col.push_back(matrices[d+1].global_index[p]);
	    }
	    M.remove_max(i);
	  }
	}
	std::reverse(col.begin(),col.end());
	M.set_col(i,col);
      }
    }
    
    for(index d=matrices.size()-1;d>=0;d--) {
      GrMatExt& M=matrices[d];
      index new_no_col=0;
      for(index i=0;i<M.get_num_cols();i++) {
	if(M.status[i]==2) { 
	  index j = M.global_index[i];
	  assert(j>=0);
	  assert(j<=i);
	  std::vector<index> col;
	  M.get_col(i,col);
	  M.set_col(j,col);
	  M.grades[j]=M.grades[i];
	  new_no_col++;
	}
      }
      index no_rows = (d==matrices.size()-1 ? matrices[d].num_rows : matrices[d+1].get_num_cols());
      M.set_dimensions(no_rows,new_no_col);
      M.grades.resize(new_no_col);
      // set_dimension does not update num_rows, even though it should...
      M.num_rows=no_rows;
      if(multi_chunk::verbose) std::cout << "d=" << d << " new dimensions are " << M.num_rows << "x" << M.get_num_cols() << std::endl;
      if(multi_chunk::verbose) std::cout << "Compare new dimensions are " << input_matrices[d].num_rows << "x" << input_matrices[d].get_num_cols() << std::endl;

      if(d<=max_d) {
	M.row_grades.clear();
	std::copy(matrices[d+1].grades.begin(),
		  matrices[d+1].grades.end(),
		  std::back_inserter(M.row_grades));
      }
    }

#if MULTI_CHUNK_TIMERS
    sparsification_timer.stop();
#endif

    long N_new=0;

    for(index i=0;i<matrices.size();i++) {
      N_new+=matrices[i].get_num_cols();
    }
    if(multi_chunk::verbose) std::cout << "Multi-chunk is done" << std::endl;

    if(multi_chunk::verbose) std::cout << "N before=" << N << std::endl;
    if(multi_chunk::verbose) std::cout << "N after =" << N_new << std::endl;
    if(multi_chunk::verbose) std::cout << "Compression rate: " << std::setprecision(4) << double(N_new)/N << std::endl;
    
  }
  
 

  template<
    typename InputStream,
    typename OutputStream,
    typename ColumnType=phat::vector_vector,
    typename CoordinateTraits=mpp_utils::Coordinate_traits_with_map<double>
    >
    void compress_stream(InputStream& instr,
			 OutputStream& outstr) {


#if MULTI_CHUNK_TIMERS
    initialize_timers();
    overall_timer.start();
#endif

    typedef mpp_utils::Graded_matrix<ColumnType, CoordinateTraits> GrMat;
    
    std::vector<GrMat> matrices;

#if MULTI_CHUNK_TIMERS
    io_timer.start();
#endif

    typedef typename GrMat::Coordinate_traits Grade_cast;
    scc::Scc<Grade_cast> parser(instr,&Grade_cast::get_instance());
    
    mpp_utils::create_graded_matrices_from_scc2020(parser,matrices);
    
#if MULTI_CHUNK_TIMERS
    io_timer.stop();
#endif


    multi_chunk::compress(matrices);


    if(multi_chunk::verbose) std::cout << "Writing output" << std::endl;
#if MULTI_CHUNK_TIMERS
    io_timer.resume();
#endif
    
    mpp_utils::print_in_scc_format(matrices,outstr);
#if MULTI_CHUNK_TIMERS
    io_timer.stop();
#endif

#if MULTI_CHUNK_TIMERS
    if(multi_chunk::verbose) print_timers();
#endif


  }
 
  template<
    typename ColumnType=phat::vector_vector,
    typename CoordinateTraits=mpp_utils::Coordinate_traits_with_map<double>
    >
    void compress(const std::string& infile,
		  const std::string& outfile) {

    std::ifstream ifstr(infile);
    std::ofstream ofstr(outfile);

    compress_stream(ifstr,ofstr);
    
    ifstr.close();
    ofstr.close();

  }
  
}
