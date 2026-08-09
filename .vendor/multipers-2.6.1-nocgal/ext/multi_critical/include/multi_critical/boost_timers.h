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

#include <iostream>
#include <boost/timer/timer.hpp>
#include <cmath>

namespace multi_critical {

    boost::timer::cpu_timer overall_timer, sorting_timer,multi_chunk_timer,transformation_to_1_critical_timer,prepare_input_complex_timer,prepare_output_complex_timer, compute_f0_timer,compute_f1_timer,compute_h0_timer,sort_by_grades_timer,convert_to_graded_matrices_timer,output_file_timer,read_input_timer,mpfree_timer,
	compute_f2_timer,compute_h1_timer,compute_hh_timer,construct_boundaries_timer,
	test_timer_1,test_timer_2,test_timer_3,test_timer_4;
  
void initialize_timers() {
  overall_timer.start();
  overall_timer.stop();
  sorting_timer.start();
  sorting_timer.stop();
  multi_chunk_timer.start();
  multi_chunk_timer.stop();
  mpfree_timer.start();
  mpfree_timer.stop();
  transformation_to_1_critical_timer.start();
  transformation_to_1_critical_timer.stop();
  prepare_input_complex_timer.start();
  prepare_input_complex_timer.stop();
  prepare_output_complex_timer.start();
  prepare_output_complex_timer.stop();
  compute_f0_timer.start();
  compute_f0_timer.stop();
  compute_f1_timer.start();
  compute_f1_timer.stop();
  compute_h0_timer.start();
  compute_h0_timer.stop();
  sort_by_grades_timer.start();
  sort_by_grades_timer.stop();
  convert_to_graded_matrices_timer.start();
  convert_to_graded_matrices_timer.stop();
  output_file_timer.start();
  output_file_timer.stop();
  read_input_timer.start();
  read_input_timer.stop();
  compute_f2_timer.start();
  compute_f2_timer.stop();
  compute_h1_timer.start();
  compute_h1_timer.stop();
  compute_hh_timer.start();
  compute_hh_timer.stop();
  construct_boundaries_timer.start();
  construct_boundaries_timer.stop();
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
    std::cout << "Overall timer:                " << double(overall_timer.elapsed().wall)/std::pow(10,9) << std::endl;
  }
  std::cout << "Read input timer:             "; pretty_print_timer(read_input_timer,total);
  std::cout << "Transformation timer:         "; pretty_print_timer(transformation_to_1_critical_timer,total);
  std::cout << "Sorting timer:                "; pretty_print_timer(sorting_timer,total);
  std::cout << "Multi chunk timer:            "; pretty_print_timer(multi_chunk_timer,total);
  std::cout << "Mpfree timer:                 "; pretty_print_timer(mpfree_timer,total);
  std::cout << "Output file timer:            "; pretty_print_timer(output_file_timer,total);
  std::cout << " ----- Subroutines of transformation -------" << std::endl;
  double transform_time=transformation_to_1_critical_timer.elapsed().wall;
  std::cout << "Prepare input complex timer:  "; pretty_print_timer(prepare_input_complex_timer,transform_time);
  std::cout << "Prepare output complex timer: "; pretty_print_timer(prepare_output_complex_timer,transform_time);
  std::cout << "Compute f0 timer:             "; pretty_print_timer(compute_f0_timer,transform_time);
  std::cout << "Compute f1 timer:             "; pretty_print_timer(compute_f1_timer,transform_time);
  std::cout << "Compute h0 timer:             "; pretty_print_timer(compute_h0_timer,transform_time);
  std::cout << "Compute f2 timer:             "; pretty_print_timer(compute_f2_timer,transform_time);
  std::cout << "Compute h1 timer:             "; pretty_print_timer(compute_h1_timer,transform_time);
  std::cout << "Compute hh timer:             "; pretty_print_timer(compute_hh_timer,transform_time);
  std::cout << "Construct boundaries timer:   "; pretty_print_timer(construct_boundaries_timer,transform_time);
  std::cout << "Sort by grades timer:         "; pretty_print_timer(sort_by_grades_timer,transform_time);
  std::cout << "Convert to gr matrices timer: "; pretty_print_timer(convert_to_graded_matrices_timer,transform_time);
  std::cout << " ----- Test timer -------" << std::endl;
  std::cout << "Test timer 1:           "; pretty_print_timer(test_timer_1,total);
  std::cout << "Test timer 2:           "; pretty_print_timer(test_timer_2,total);
  std::cout << "Test timer 3:           "; pretty_print_timer(test_timer_3,total);
  std::cout << "Test timer 4:           "; pretty_print_timer(test_timer_4,total);
}

}
  
