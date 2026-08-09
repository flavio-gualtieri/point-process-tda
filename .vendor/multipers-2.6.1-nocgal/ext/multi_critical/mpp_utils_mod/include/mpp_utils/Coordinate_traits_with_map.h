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

#include<unordered_map>

namespace mpp_utils {

  // The purpose of this class is: If double values are read from
  // an input file, it is made sure that the values are written
  // to the output file in exactly the same way

  template<typename Coordinate_> class Coordinate_traits_with_map {
    
  public:
    
    typedef Coordinate_ Coordinate;
    typedef Coordinate_traits_with_map<Coordinate> Self;
    
    typedef Coordinate value_type;

  private:
    
    Coordinate_traits_with_map() {}

  public:

    static Self& get_instance() { 
      static Self _instance;
      return _instance;
    }
    
    std::unordered_map<Coordinate,std::string> coor_map;

    Coordinate from_string(char* str) {
      Coordinate result = Coordinate(atof(str));
      coor_map[result]=std::string(str);
      //std::cout << "Mapping " << result << " to " << str << std::endl;
      //std::cout << "Ma size" << coor_map.size() << std::endl;
      return result;
    }

    // Needed for compatibility with SCC
    Coordinate operator() (const std::string& str) {
      return from_string((char*)str.c_str());
    }

    void to_stream(std::ostream& out, Coordinate& c) {
      //std::cout << "Looking for " << c << " in map of size " << coor_map.size() << std::endl;
      if(coor_map.find(c)!=coor_map.end()) {
	out <<  coor_map[c];
      } else {
	//std::cout << "Not found " << c << std::endl;
	out << c;
      }
    }
    
    struct Compare {
      
      bool operator() (const Coordinate& a, const Coordinate& b) {
	return a<b;
      }

    };
  };

} // of namespace mpp_utils
