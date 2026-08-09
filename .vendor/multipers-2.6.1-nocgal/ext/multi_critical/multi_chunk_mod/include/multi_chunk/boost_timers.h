/* Copyright 2020 TU Graz
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

#include <iostream>
#include <boost/timer/timer.hpp>
#include <cmath>

namespace multi_chunk {


  boost::timer::cpu_timer overall_timer, io_timer, local_reduction_timer, sparsification_timer;
  
  void initialize_timers() {
    overall_timer.start();
    overall_timer.stop();
    io_timer.start();
    io_timer.stop();
    local_reduction_timer.start();
    local_reduction_timer.stop();
    sparsification_timer.start();
    sparsification_timer.stop();
  }

  void print_timers(double total=double(overall_timer.elapsed().wall), bool print_overall=true) {
    if(print_overall) {
      std::cout << "Overall timer:         " <<  double(overall_timer.elapsed().wall)/std::pow(10,9) << std::endl;
    }
    std::cout << "IO timer:              " << double(io_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(io_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    std::cout << "Local reduction timer: " << double(local_reduction_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(local_reduction_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    std::cout << "Sparsification timer:  " << double(sparsification_timer.elapsed().wall)/std::pow(10,9) <<  "     ( " << double(sparsification_timer.elapsed().wall)/total*100 << "% )" << std::endl;
  }

  
  
}
