/* Copyright 2021 TU Graz
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

#include <iostream>
#include <boost/timer/timer.hpp>
#include <cmath>

namespace mpfree {

  boost::timer::cpu_timer overall_timer, chunk_timer,io_timer, mingens_timer, kerbasis_timer, 
    reparam_timer,minimize_timer,syzygy_timer;
  
  void initialize_timers() {
    overall_timer.start();
    overall_timer.stop();
    chunk_timer.start();
    chunk_timer.stop();
    io_timer.start();
    io_timer.stop();
    mingens_timer.start();
    mingens_timer.stop();
    kerbasis_timer.start();
    kerbasis_timer.stop();
    reparam_timer.start();
    reparam_timer.stop();
    minimize_timer.start();
    minimize_timer.stop();
    syzygy_timer.start();
    syzygy_timer.stop();
  }

  void print_timers(double total=double(overall_timer.elapsed().wall),bool print_overall=true, bool print_syzygy_timer=false) {
    if(print_overall) {
      std::cout << "Overall timer: " << double(overall_timer.elapsed().wall)/std::pow(10,9) << std::endl;
    }
    std::cout << "IO timer:              " << double(io_timer.elapsed().wall)/std::pow(10,9) << "     ( "  <<  double(io_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    std::cout << "Chunk timer:           " << double(chunk_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(chunk_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    std::cout << "Mingens timer:         " << double(mingens_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(mingens_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    std::cout << "Kerbasis timer:        " << double(kerbasis_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(kerbasis_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    std::cout << "Reparam timer:         " << double(reparam_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(reparam_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    std::cout << "Minimize timer:        " << double(minimize_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(minimize_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    if(print_syzygy_timer) {
      std::cout << "Syzygy timer:          " << double(syzygy_timer.elapsed().wall)/std::pow(10,9) << "     ( " << double(syzygy_timer.elapsed().wall)/total*100 << "% )" << std::endl;
    }
  }
}
  
