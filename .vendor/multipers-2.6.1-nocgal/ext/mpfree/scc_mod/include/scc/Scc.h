#pragma once

#include<cassert>
#include<vector>
#include<string>
#include<iostream>
#include<sstream>
#include<fstream>

#include<scc/basic.h>
#include<scc/Column_data.h>

namespace scc {


  struct Cast_to_double {
    typedef double value_type;
    double operator() (const std::string& str) {
      return double(stod(str));
    }
  };

  template<typename T> struct Generic_cast {
    typedef T value_type;
    T operator() (const std::string& str) {
      return static_cast<T>(str);
    }
  };

  struct Cast_to_int {
    typedef int value_type;
    double operator() (const std::string& str) {
      return stoi(str);
    }
  };
  
  // Removes comments and empty lines
  bool get_next_line(std::istream& istr, 
		     std::string& next,
		     bool throw_error_on_fail=true) {

    while(istr.good()) {

      std::getline(istr,next);
      //std::cout << "Next: " << next << std::endl;
      // Remove everything after first '#'
      std::size_t pos_of_comment=next.find_first_of('#');
      next = next.substr(0,pos_of_comment);
      
      // Remove starting blanks
      std::size_t pos_of_first_non_blank=next.find_first_not_of(' ');
      if(pos_of_first_non_blank==std::string::npos) {
	pos_of_first_non_blank=next.length();
      }
      next = next.substr(pos_of_first_non_blank,std::string::npos);
      
      // Remove ending blanks
      std::size_t pos_of_last_non_blank=next.find_last_not_of(' ');
      if(pos_of_last_non_blank!=std::string::npos) {
	next = next.substr(0,pos_of_last_non_blank+1);
      }
      
      if(next.size()>0) {
	//std::cout << "At the end: " << next << std::endl;
	return true;
      }
    }
    if(throw_error_on_fail) {
      throw ParseError("Unexpected end of file");
    }
    return false;
  }
    
  // Method does not change the position in the stream!
  std::string get_magic_word(std::istream& istr) {
    
    int pos = istr.tellg();
    std::string next="";
    get_next_line(istr,next,false);
    istr.clear();
    istr.seekg(pos);
    return next;
  }
    

  std::string get_magic_word(std::string filename) {
    std::ifstream ifstr(filename);
    std::string next=get_magic_word(ifstr);
    ifstr.close();
    return next;
  }

  struct In_memory_level_data {
    int _number_of_generators;
    int _number_of_parameters;
    int curr_index;
    std::vector<Column_data> column_data;
    
    In_memory_level_data(std::istream& istr,
			 int number_of_generators,
			 int number_of_parameters)
      : _number_of_generators(number_of_generators),
	_number_of_parameters(number_of_parameters),
	curr_index(0) {
    }
    
    bool has_next_column() {
      return this->curr_index<this->_number_of_generators;
    }

    Column_data& next_column() {
      return this->column_data[curr_index++];
    }

    void reset() {
      this->curr_index=0;
    }

    int number_of_generators() {
      return this->_number_of_generators;
    }

    bool read_next_column(std::istream& istr,
			  bool allow_eof=false,
			  bool store_boundary=true) {
      std::string next;
      bool line_found= get_next_line(istr,next,!allow_eof);
      if(!line_found && allow_eof) {
	return false;
      }
      Column_data col;
      parse_line_to_column_data(next,this->_number_of_parameters,col);
      if(!store_boundary) {
	col.boundary.clear();
      }
      this->column_data.push_back(col);
      return true;
    }

    void clear() {
      column_data.clear();
      column_data.shrink_to_fit();
    }
      

  };

  struct From_file_level_data {
    int _number_of_generators;
    int _number_of_parameters;
    int curr_index;
    std::istream& istr;
    std::streampos curr_pos;
    std::streampos start_pos;
    
    From_file_level_data(std::istream& istr,
			 int number_of_generators,
			 int number_of_parameters)
      : _number_of_generators(number_of_generators),
	_number_of_parameters(number_of_parameters),
	curr_index(0),
	istr(istr),
	curr_pos(-1),
	start_pos(-1) {
    }
    
    bool has_next_column() {
      return this->curr_index<this->_number_of_generators;
    }

    Column_data next_column() {
      istr.clear();
      std::streampos old_streampos=istr.tellg();
      //std::cout << "Curr pos " << curr_pos << std::endl;
      istr.seekg(this->curr_pos);
      
      std::string next;
      get_next_line(istr,next);
      
      Column_data col;
      parse_line_to_column_data(next,this->_number_of_parameters,col);

      this->curr_pos=istr.tellg();
      curr_index++;
      istr.seekg(old_streampos);
      return col;
    }

    void reset() {
      this->curr_index=0;
      this->curr_pos=start_pos;
    }

    int number_of_generators() {
      return this->_number_of_generators;
    }

    bool read_next_column(std::istream& istr, 
			  bool allow_eof=false,
			  bool allow_boundary=true) {
      if(this->start_pos==-1) {
	this->start_pos=istr.tellg();
	curr_pos=start_pos;
      }
      std::string next;
      bool line_found= get_next_line(istr,next,!allow_eof);
      if(!line_found && allow_eof) {
	return false;
      }
      return true;
    }

    void clear() {
    }

  };


  template<
    typename GradeCast=Cast_to_double,
    typename FieldCast=Cast_to_int,
    typename LevelData=In_memory_level_data
    >    
  class Scc {

    public:

    typedef GradeCast Grade_cast;
    typedef FieldCast Field_cast;
    typedef LevelData Level_data;
    

    typedef typename Grade_cast::value_type Grade;
    typedef typename Field_cast::value_type Field;

    protected:
    
    Grade_cast* grade_cast;
    Field_cast* field_cast;

    std::vector<bool> _parameters;
    
    std::vector<Level_data> _levels;

    bool _grades_on_last_level;

    public:

    Scc(std::istream& istr,
	Grade_cast* grade_cast_= new Grade_cast(), 
	Field_cast* field_cast= new Field_cast()) 
    : grade_cast(grade_cast_), field_cast(field_cast) {
      
      std::string next;

      {
	// 1.Find keyword
	get_next_line(istr,next);
	
	if(next!="scc2020") {
	  throw ParseError("Keyword scc2020 expected");
	}
      }

      {
	// 2.Read number of parameters
	get_next_line(istr,next);
	try {
	  std::size_t pos;
	  int no_parameters=stoi(next,&pos);
	  if(pos!=next.size()) {
	    throw ParseError();
	  }
	  for(int i=0;i<no_parameters;i++) {
	    this->_parameters.push_back(true);
	  }
	} catch(...) {
	  throw ParseError("Integer expected in line "+next);
	}
      }

      int d = number_of_parameters();

      {
	// 3.Read number of generators
	get_next_line(istr,next);
	
	std::stringstream sstr(next);
	while(sstr.good()) {
	  int number;
	  sstr >> number;
	  if(scc::verbose) std::cout << "Number of Generators " << this->_levels.size() << ": " << number << std::endl;
	  this->_levels.push_back(Level_data(istr,number,d));
	}
      }

      {
	
	// 4.Try to read (optional) reverse vector
	std::streampos curr_streampos=istr.tellg();
	get_next_line(istr,next);

	if(next.rfind("--reverse",0)==0) {
	  std::string next_sub=next.substr(9);
	  std::stringstream sstr(next_sub);
	  while(sstr.good()) {
	    int number;
	    sstr >> number;
	    if(number<1 || number>this->_parameters.size()) {
	      throw ParseError("Integer unexpected in "+next);
	    }
	    std::cout << "Parameter " << number << " reversed" << std::endl;
	    this->_parameters[number-1]=false;
	  }
	} else {
	  istr.seekg(curr_streampos);
	}
      }

      int n = number_of_levels()-1;

      {
	// 5.Read the matrices      
	for(int lev=0;lev<n;lev++) {
	  for(int gen=0;gen<this->_levels[lev].number_of_generators();gen++) {
	    this->_levels[lev].read_next_column(istr);
	  }
	  this->_levels[lev].reset();
	}
      }

      {

	// 6. Read optional grades on the last level
	int count_last_level=0;
	while(this->_levels[n].read_next_column(istr,true,false)) {
	  //std::cout << "Next line (addendum) " << next << std::endl;
	  if(count_last_level >= this->_levels[n].number_of_generators()) {
	    throw ParseError("Unexpected line "+next);
	  }
	  count_last_level++;
	}
	if(count_last_level>0 && count_last_level < this->_levels[n].number_of_generators()) {
	  throw ParseError("Unexpected end of file");
	}
	this->_grades_on_last_level=(count_last_level>0);
      }
    }

    int number_of_parameters() {
      return this->_parameters.size();
    }

    int number_of_levels() {
      return this->_levels.size();
    }

    bool has_next_column(int level) {
      assert(level>=1 && level<=number_of_levels());
      return this->_levels[level-1].has_next_column();
    }
    
    bool has_grades_on_last_level() {
      return this->_grades_on_last_level;
    }

    template<
      typename OutputIterator1,
      typename OutputIterator2
            >
    void next_column(int level, 
		     OutputIterator1 out1,
		     OutputIterator2 out2) {
      Column_data col = this->_levels[level-1].next_column();
      for(auto grade : col.grades) {
	out1++=(*grade_cast)(grade);
      }
      for(auto bd : col.boundary) {
	out2++=std::make_pair(bd.first,(*field_cast)(bd.second));
      }
    }

    int number_of_generators(int level) {
      return this->_levels[level-1].number_of_generators();
    }

    void reset(int level) {
      this->_levels[level-1].reset();
    }

    void clear() {
      for(int i=0;i<this->_levels.size();i++) {
	this->_levels[i].clear();
      }
    }

  };



} //of namespace scc
