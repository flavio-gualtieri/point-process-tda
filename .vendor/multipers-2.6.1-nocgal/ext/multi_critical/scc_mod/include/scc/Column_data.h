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
	
	bool semicolon_found=(s.find(";")!=std::string::npos);
	out.grades.clear();
	out.boundary.clear();
	int pos=0,old_pos=0;
	if(! semicolon_found) {
	    // The input must be 1-critical - read d entries as grades
	    while(s[pos]==' ') { 
		pos++;
	    }
	    old_pos=pos;
	    for(int i=0;i<d;i++) {
		while(s[pos]!=' ') { 
		    pos++;
		}
		out.grades.push_back(s.substr(old_pos,pos-old_pos));
		while(s[pos]==' ') { 
		    pos++;
		}
		old_pos=pos;
	    }
	} else {
	    // Input might be multi-critical - just read grades
	    // until the semicolon - the split into the particular grades
	    // is not down by this class (mostly for backward compatibility)
	    while(s[pos]==' ') { 
		pos++;
	    }
	    old_pos=pos;
	    while(s[pos]!=';') {
		while(s[pos]!=' ' && s[pos]!=';') { 
		    pos++;
		}
		out.grades.push_back(s.substr(old_pos,pos-old_pos));
		while(s[pos]==' ') { 
		    pos++;
		}
		old_pos=pos;
	    }
	    pos++;
	    if(s.find(";",pos)!=std::string::npos) {
		std::cerr << "Found too many ';' in line" << std::endl;
		std::exit(1);
	    }
	}
	while(pos<s.length() && s[pos]==' ') {
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
