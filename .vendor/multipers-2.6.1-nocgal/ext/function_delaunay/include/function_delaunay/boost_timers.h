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

#include <iostream>
#include <boost/timer/timer.hpp>
#include <cmath>

boost::timer::cpu_timer overall_timer, complex_timer,face_timer,meb_timer,graded_matrices_timer,multi_chunk_timer,mpfree_timer,file_output_timer,delaunay_timer,initial_timer,
  test_timer_1,test_timer_2,test_timer_3,test_timer_4;
  
void initialize_timers() {
  overall_timer.start();
  overall_timer.stop();
  initial_timer.start();
  initial_timer.stop();
  complex_timer.start();
  complex_timer.stop();
  face_timer.start();
  face_timer.stop();
  meb_timer.start();
  meb_timer.stop();
  graded_matrices_timer.start();
  graded_matrices_timer.stop();
  multi_chunk_timer.start();
  multi_chunk_timer.stop();
  mpfree_timer.start();
  mpfree_timer.stop();
  file_output_timer.start();
  file_output_timer.stop();
  delaunay_timer.start();
  delaunay_timer.stop();
  test_timer_1.start();
  test_timer_1.stop();
  test_timer_2.start();
  test_timer_2.stop();
  test_timer_3.start();
  test_timer_3.stop();
  test_timer_4.start();
  test_timer_4.stop();
  
}

void pretty_print_timer(boost::timer::cpu_timer& timer,double total) {
  if(double(timer.elapsed().wall)/std::pow(10,9)<0.0001) {
    std::cout << "0" << std::endl;
  } else{
    std::cout << double(timer.elapsed().wall)/std::pow(10,9) << "     ( "  <<  double(timer.elapsed().wall)/total*100 << "% )" << std::endl;
  }
}

void print_timers(double total=double(overall_timer.elapsed().wall),bool print_overall=true) {
  if(print_overall) {
    std::cout << "Overall timer: " << double(overall_timer.elapsed().wall)/std::pow(10,9) << std::endl;
  }
  std::cout << "Inital timer:           "; pretty_print_timer(initial_timer,total);
  std::cout << "Complex timer:          "; pretty_print_timer(complex_timer,total);
  std::cout << "Face timer:             "; pretty_print_timer(face_timer,total);
  std::cout << "Meb timer:              "; pretty_print_timer(meb_timer,total);
  std::cout << "Graded matrices timer:  "; pretty_print_timer(graded_matrices_timer,total);
  std::cout << "Multi chunk timer:      "; pretty_print_timer(multi_chunk_timer,total);
  std::cout << "Mpfree timer:           "; pretty_print_timer(mpfree_timer,total);
  std::cout << "File output timer:      "; pretty_print_timer(file_output_timer,total);
  std::cout << "Delaunay timer:         "; pretty_print_timer(delaunay_timer,total);
  std::cout << "Test timer 1:           "; pretty_print_timer(test_timer_1,total);
  std::cout << "Test timer 2:           "; pretty_print_timer(test_timer_2,total);
  std::cout << "Test timer 3:           "; pretty_print_timer(test_timer_3,total);
  std::cout << "Test timer 4:           "; pretty_print_timer(test_timer_4,total);
}

  
