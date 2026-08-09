#pragma once

#include<vector>
#include<string>
#include<iostream>

#include<scc/basic.h>

namespace scc {

  struct Column_data {
    std::vector<std::string> grades;
    std::vector< std::pair<index,std::string> > boundary;
  };
  
  void parse_line_to_column_data(const std::string& s,
				 int d,
				 Column_data& out) {
    out.grades.clear();
    out.boundary.clear();
    int pos=0;
    while(s[pos]==' ') { 
      pos++;
    }
    int old_pos=pos;
    for(int i=0;i<d-1;i++) {
      while(s[pos]!=' ') { 
	pos++;
      }
      out.grades.push_back(s.substr(old_pos,pos-old_pos));
      while(s[pos]==' ') { 
	pos++;
      }
      old_pos=pos;
    }
    // For pos d, we need special treatment because of the optional semicolon
    bool semicolon_found=false;
    while(s[pos]!=' ' && s[pos]!=';') { 
      pos++;
    }
    out.grades.push_back(s.substr(old_pos,pos-old_pos));
    if(s[pos]==';') {
      semicolon_found=true;
      pos++;
    }
    while(pos<s.length() && (s[pos]==' ' || s[pos]==';')) {
      if(s[pos]==';') {
	if(semicolon_found) {
	  std::cerr << "Found too many ';' in line" << std::endl;
	  std::exit(1);
	} else {
	  semicolon_found=true;
	}
      }
      pos++;
    }
    old_pos=pos;
    while(pos<s.length()) {
      while(pos<s.length() && (s[pos]>='0' && s[pos]<='9')) {
	pos++;
      }
      int next_index = atoi(s.substr(old_pos,pos-old_pos).c_str());

      std::string next_coeff;
      if(pos==s.length() || s[pos]==' ') {
	next_coeff="1";
      } else if(s[pos]==':') {
	pos++;
	old_pos=pos;
	while(pos<s.length() && (s[pos]!=' ')) {
	  pos++;
	}
	next_coeff = s.substr(old_pos,pos-old_pos);
      } else {
	std::cerr << "Unexpected character!" << std::endl;
	std::exit(1);
      }
      out.boundary.push_back(std::make_pair(next_index,next_coeff));
      while(pos<s.length() && s[pos]==' ') {
	pos++;
      }
      old_pos=pos;
    }
  }
  
    
      


} // of namespace scc
