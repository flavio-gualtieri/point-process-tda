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

#include<multi_critical/basic.h>

namespace multi_critical {

    struct Grade {
	double x;
	double y;
	Grade(double x,double y) : x(x), y(y) {}
	Grade() {}
    };
    
    Grade join (Grade& g1,Grade& g2) {
	return Grade(std::max(g1.x,g2.x),std::max(g1.y,g2.y));
    }
    
    bool operator<= (Grade& g1,Grade& g2) {
	return g1.x <= g2.x && g1.y <= g2.y;
    }
    
    bool are_comparable(Grade& g1, Grade& g2) {
	return g1<=g2 || g2<=g1;
    }

    struct Lex_sort_grade {
	
	bool operator() (Grade& g1,Grade& g2) {
	    if(g1.x<g2.x) {
		return true;
	    }
	    if(g1.x>g2.x) {
		return false;
	    }
	    return g1.y<g2.y;
	}
	
    };

#if 1
    void get_incomparable_grades_in_lex_order(std::vector<double>& raw_grades,
					      std::vector<Grade>& result) {
	
	if(raw_grades.size()%2!=0) {
	    std::cerr << "Odd number of grades found!" << std::endl;
	    return;
	}
	if(raw_grades.size()==0) {
	    std::cerr << "No grades found!" << std::endl;
	    return;
	}
	std::vector<Grade> grades;
	
	for(int i=0;i<raw_grades.size();i+=2) {
	    Grade gr(raw_grades[i],raw_grades[i+1]);
	    grades.push_back(gr);
	}
	std::sort(grades.begin(),grades.end(),Lex_sort_grade());
	result.push_back(grades[0]);
	int last_pushed=0;
	for(int i=1;i<grades.size();i++) {
	    if(!are_comparable(grades[last_pushed],grades[i])) {
		result.push_back(grades[i]);
		last_pushed=i;
	    }
	}
	//std::cout << "Obtained " << raw_grades.size()/2 << " bigrades, result is " << result.size() << std::endl;
    }

#else
        void get_incomparable_grades_in_lex_order(std::vector<double>& raw_grades,
					      std::vector<Grade>& result) {
	
	if(raw_grades.size()%2!=0) {
	    std::cerr << "Odd number of grades found!" << std::endl;
	    return;
	}
	if(raw_grades.size()==0) {
	    std::cerr << "No grades found!" << std::endl;
	    return;
	}
		
	for(int i=0;i<raw_grades.size();i+=2) {
	    Grade gr(raw_grades[i],raw_grades[i+1]);
	    result.push_back(gr);
	}
    }
    
    
#endif
    
    index get_copy_smaller_equal_than(std::vector<Grade>& grades, Grade& g) {
	//std::cout << "Grade: " << g.x << " " << g.y << std::endl;
#if 0
	for(int i=0;i<grades.size();i++) {
#else
        for(int i=grades.size()-1;i>=0;i--) {
#endif
	    //std::cout << "Candidate: " << grades[i].x << " " << grades[i].y << std::endl;
	    if(grades[i]<=g) {
		//std::cout << "Works" << std::endl;
		return i;
	    }
	}
	std::cerr.precision(std::numeric_limits<double>::max_digits10 - 1);
	std::cerr << "Warning, no smaller grade found for " << g.x << " " << g.y << "!" << std::endl;
	std::cerr << "Candidates were: ";
	for(auto& gr: grades) {
	    std::cerr << gr.x << " " << gr.y << ", ";
	}
	std::cerr << std::endl;
	return -1;
    }
		
    


    
} // of namespace multi_critical
