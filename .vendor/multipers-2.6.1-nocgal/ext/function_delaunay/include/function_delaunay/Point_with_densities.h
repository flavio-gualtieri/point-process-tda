/* Copyright 2023 TU Graz
   Author: Michael Kerber
   
   This file is part of function_delaunay
   
   function_delaunay is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   function_delaunay is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with function_delaunay.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include<vector>

namespace function_delaunay {

    struct Point_with_densities {
	std::vector<double> x;
	std::vector<double> densities;
	// This is used internally for convenience
	int idx;
	
	template<typename Iterator>
	Point_with_densities(Iterator begin, Iterator end,double density) {
	    std::copy(begin,end,std::back_inserter(x));
	    densities.push_back(density);
	}
	template<typename Iterator>
	Point_with_densities(Iterator begin, Iterator end,double density1,double density2) {
	    std::copy(begin,end,std::back_inserter(x));
	    densities.push_back(density1);
	    densities.push_back(density2);
	}
	template<typename CoorIterator,typename DensityIterator>
	Point_with_densities(CoorIterator coors_begin, CoorIterator coors_end,DensityIterator densities_begin,DensityIterator densities_end) {
	    std::copy(coors_begin,coors_end,std::back_inserter(x));
	    std::copy(densities_begin,densities_end,std::back_inserter(densities));
	}
	int dimension() const {
	    return x.size();
	}
	int number_of_densities() const {
	    return densities.size();
	}
    };
  
    struct Lex_sort_by_density {
	bool operator() (Point_with_densities& a, Point_with_densities& b) {
	    
	    bool result=false;
	    for(int i=0;i<a.number_of_densities();i++) {
		if(a.densities[i]<b.densities[i]) {
		    result=true;
		    break;
		}
		if(a.densities[i]>b.densities[i]) {
		    result=false;
		    break;
		}
	    }
	    return result;
	}
    };

    struct CoLex_sort_by_density {
        bool operator() (Point_with_densities& a, Point_with_densities& b) {

            bool result = false;
            for (int i = a.number_of_densities() - 1; i >= 0; --i) {
                if (a.densities[i] < b.densities[i]) {
                    result = true;
                    break;
                }
                if (a.densities[i] > b.densities[i]) {
                    result = false;
                    break;
                }
            }
            return result;
        }
    };
}
