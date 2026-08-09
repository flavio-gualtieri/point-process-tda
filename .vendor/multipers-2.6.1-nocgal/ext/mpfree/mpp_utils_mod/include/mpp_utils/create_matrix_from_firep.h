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
#include<mpp_utils/Pre_column_struct.h>

namespace mpp_utils {


  struct File_reader {
    
    std::size_t pos;
    
    std::string data;

    

    //Actually, with this constructor, it is rather a "Stream_reader", but it is needed for compatibility
    File_reader(std::istream& istr) : pos(0) {
      // Adapted from https://stackoverflow.com/questions/3203452/how-to-read-entire-stream-into-a-stdstring
      char buffer[4096];
      while (istr.read(buffer, sizeof(buffer))) {
        data.append(buffer, sizeof(buffer));
      }
      data.append(buffer, istr.gcount());
    }
    

    
    File_reader(const char *filename) : pos(0) {
      // Adapted code from http://insanecoding.blogspot.com/2011/11/how-to-read-in-file-in-c.html
      std::FILE *fp = std::fopen(filename, "rb");
      if (fp) {
	std::fseek(fp, 0, SEEK_END);
	data.resize(std::ftell(fp));
	std::rewind(fp);
	// Return value is stored to supress compiler warning
	std::size_t no_read=std::fread(&data[0], 1, data.size(), fp);
	if(verbose) std::cout << "Read " << no_read << " characters..." << std::flush;
	std::fclose(fp);
      } else {
	throw(errno);
      }
    }
    
    std::string next_line() {
      std::size_t old_pos=pos;
      while(data[pos]!='\n') {
	pos++;
      }
      if(pos!=data.length()) {
	pos++;
      }
      return data.substr(old_pos,pos-old_pos-1);
    }
  };
 
  // The sign is just a hack for the perturbation scenario and normally has no effect
  template<typename GradedMatrix, typename Pre_column >
  void load_prematrix_contents(File_reader& reader,
			       std::vector<Pre_column>& pre_matrix,
			       int n,
			       int sign) {

  
#if 1

    pre_matrix.resize(n);
    std::vector<std::string> lines;
    lines.resize(n);
    //std::string line;
    for(int i=0;i<n;i++) {
      lines[i]=reader.next_line();
    }
    
    typename GradedMatrix::Coordinate_traits& traits 
      = GradedMatrix::Coordinate_traits::get_instance();

#if STRINGTOK_R_AVAILABLE 
#pragma omp parallel for schedule(guided,1)
#endif
    for(int i=0;i<n;i++) {
      Pre_column& curr = pre_matrix[i];
      curr.idx=i;
      char* line = (char*)lines[i].c_str();
#if STRINGTOK_R_AVAILABLE
      char* saveptr;
      char* token = strtok_r(line," ",&saveptr);
#else
      char* token = strtok(line," ");
#endif
      
      curr.grade.at.push_back(traits.from_string(token));
      //std::cout << "value: " <<  curr.grade.first_val << std::endl;
#if STRINGTOK_R_AVAILABLE
      token = strtok_r(NULL," ",&saveptr);
#else
      token = strtok(NULL," ");
#endif
      curr.grade.at.push_back(traits.from_string(token));
#if STRINGTOK_R_AVAILABLE
      token = strtok_r(NULL," ",&saveptr);
#else
      token = strtok(NULL," ");
#endif
      if(strcmp(token,";")) {
	std::cerr << "Semicolon missing" << std::endl;
      }
      std::vector<index> indices;
#if STRINGTOK_R_AVAILABLE
      while(token = strtok_r(NULL," ",&saveptr) ) {
#else
      while(token = strtok(NULL," ") ) {
#endif
	curr.boundary.push_back(atoi(token));
      }
      std::sort(curr.boundary.begin(),curr.boundary.end());
      //std::cout << "Read " << indices.size() << " indices" << std::endl;
    }
#else
    std::string line,next;           
    for(int i=0;i<n;i++) {
      std::getline(instr,line);
      std::stringstream sstr(line);
      Coordinate x,y;
      sstr >> x >> y;
      Grade grade(x,y);
      sstr >> next;
      if(next!=";") {
	std::cerr << "Semicolon missing" << std::endl;
      }
      std::vector<index> indices;
      int next_id;
      while(sstr.good()) {
	sstr >> next_id;
	indices.push_back(next_id);
      }
      assert(grade.at.size()==2);
      pre_matrix.push_back(Pre_column(i,grade,indices));
    }
#endif
  }
    
    
  template<typename GradedMatrix,typename Pre_column>   
  void load_data_into_prematrix(File_reader& reader,
				std::vector<Pre_column>& pre_matrix1,
				std::vector<Pre_column>& pre_matrix2,
				int& r) {
    
    std::string next=reader.next_line();
    if(next!="firep") {
      std::cerr << "Keyword 'firep' expected" << std::endl;
      std::exit(1);
    }
    std::string line=reader.next_line();
    line=reader.next_line();
    //std::cout << line << std::endl;

    int t,s;

    {
      line = reader.next_line();
      std::stringstream sstr(line);
      sstr >> t >> s >> r;
    }

    if(verbose) std::cout << "t,s,r=" << t << " " << s << " " << r << std::endl;
    
    load_prematrix_contents<GradedMatrix,Pre_column>(reader,pre_matrix1,t,+1);
    load_prematrix_contents<GradedMatrix,Pre_column>(reader,pre_matrix2,s,-1);

    }
  
 
    

  template<typename GradedMatrix>
    void create_matrix_from_firep(File_reader& reader, 
				  GradedMatrix& matrix1, 
				  GradedMatrix& matrix2) {

    typedef typename GradedMatrix::Grade Grade;
    typedef Pre_column_struct<Grade> Pre_column;
    std::vector<Pre_column> pre_matrix1, pre_matrix2;
    int r;
    
    //std::cout << "Memory after file reader " << mem_info() << std::endl;
    load_data_into_prematrix<GradedMatrix,Pre_column>(reader,pre_matrix1,pre_matrix2,r);
    //std::cout << "Memory after load_Data-into_prematrix " << mem_info() << std::endl;
    Sort_pre_column_colex<Pre_column,typename GradedMatrix::Coordinate_traits::Compare> sort_pre_column_colex;
    std::sort(pre_matrix1.begin(),pre_matrix1.end(),sort_pre_column_colex);
    std::sort(pre_matrix2.begin(),pre_matrix2.end(),sort_pre_column_colex);
    //std::cout << "Memory after sorting " << mem_info() << std::endl;
    std::vector<index> re_index;
    re_index.resize(pre_matrix2.size());
    for(index i=0;i<pre_matrix2.size();i++) {
      re_index[pre_matrix2[i].idx]=i;
    }

    for(index i=0;i<pre_matrix1.size();i++) {
      for(index j=0;j<pre_matrix1[i].boundary.size();j++) {
	pre_matrix1[i].boundary[j]=re_index[pre_matrix1[i].boundary[j]];
      }
      std::sort(pre_matrix1[i].boundary.begin(),
		pre_matrix1[i].boundary.end());
    } 
    //std::cout << "Memory after re-indexing " << mem_info() << std::endl;
    {
      int n = pre_matrix1.size();
      int m = pre_matrix2.size();
      matrix1.set_dimensions(m,n);
      matrix1.grades.resize(n);

      matrix1.sync();
#pragma omp parallel for schedule(guided,1)
      for(int i=0;i<n;i++) {
	Pre_column& pcol = pre_matrix1[i];
        matrix1.grades[i]=pcol.grade;
	matrix1.set_col(i,pcol.boundary);
      }
      pre_matrix1.clear();
      pre_matrix1.shrink_to_fit();
      matrix1.sync();
      matrix1.num_rows=pre_matrix2.size();
      matrix1.number_of_parameters=2;
    }
    //std::cout << "Memory after assigning first matrix " << mem_info() << std::endl;
    {
      int n = pre_matrix2.size();
      matrix2.set_dimensions(r,n);
      matrix2.grades.resize(n);
      matrix2.sync();
#pragma omp parallel for schedule(guided,1)
      for(int i=0;i<n;i++) {
	Pre_column& pcol = pre_matrix2[i];
        matrix2.grades[i]=pcol.grade;
	matrix2.set_col(i,pcol.boundary);
      }
      pre_matrix2.clear();
      pre_matrix2.shrink_to_fit();
      matrix2.sync();
      matrix2.num_rows=r;
      matrix2.number_of_parameters=2;
    }
    //std::cout << "Memory after assigning 2nd matrix " << mem_info() << std::endl;
    assign_grade_indices_of_pair(matrix1,matrix2);
    /*
    for(index i=0;i<matrix1.num_rows;i++) {
      matrix1.row_grades.push_back(matrix2.grades[i]);
    }
    */

    //std::cout << "Memory at the end of firep proc " << mem_info() << std::endl;
  }
  
  template<typename GradedMatrix>
    void create_matrix_from_firep(const std::string& filename, 
				  GradedMatrix& matrix1, 
				  GradedMatrix& matrix2) {
    
    if(verbose) std::cout << "Loading data from file into string..." << std::flush;
    File_reader reader(filename.c_str());
    if(verbose) std::cout << "done" << std::endl;
    create_matrix_from_firep(reader, matrix1,matrix2);
  }
  
  template<typename GradedMatrix>
    void create_matrix_from_firep(std::istream& istr, 
				  GradedMatrix& matrix1, 
				  GradedMatrix& matrix2) {
    
    if(verbose) std::cout << "Loading data from stream into string..." << std::flush;
    File_reader reader(istr);
    if(verbose) std::cout << "done" << std::endl;
    create_matrix_from_firep(reader, matrix1,matrix2);
#if !NDEBUG
    check_grade_sanity(matrix1);
#endif

  }
  
}
