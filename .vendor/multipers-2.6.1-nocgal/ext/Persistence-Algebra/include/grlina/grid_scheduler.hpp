/**
 * @file grid_scheduler.hpp
 * @author Michael Kerber - adapted to graded_linalg by Jan Jendrysiak
 * @brief This file was taken from the mpfree package
   
   mpfree is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   mpfree is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with mpfree.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#ifndef GRID_SCHEDULER_HPP
#define GRID_SCHEDULER_HPP

#include <queue>
#include <map>
#include <limits>
#include <cassert>
#include <utility>

namespace graded_linalg {

  template<typename index>
  struct Sort_grades {
      using index_pair = std::pair<index, index>;
  
      bool operator()(const index_pair& c1, const index_pair& c2) const noexcept {
          return (c1.first > c2.first) || (c1.first == c2.first && c1.second > c2.second);
      }
  };
  
  template<typename index>
  class Grid_scheduler {
      using index_pair = std::pair<index, index>;
      
      private:
      std::priority_queue<index_pair, std::vector<index_pair>, Sort_grades<index>> grades;
      std::map<index_pair, index_pair> index_range;
      index_pair curr_grade = {std::numeric_limits<index>::min(), std::numeric_limits<index>::min()};

      public:
          Grid_scheduler() = default;
      
          // Template constructor for GradedMatrix
          template<typename GradedMatrix>
          explicit Grid_scheduler(GradedMatrix& M) {
              initialize(M);
          }
      
          int size() const noexcept {
              return grades.size();
          }
      
          bool at_end() const noexcept {
              return grades.empty();
          }
      
          index_pair next_grade() noexcept {
              if (grades.empty()) {
                  return {0, 0};
              }
      
              index_pair result = std::move(grades.top());
              grades.pop();
      
              // Skip duplicate entries
              while (!grades.empty() && grades.top() == result) {
                  grades.pop();
              }
      
              curr_grade = result;
              return result;
          }
      
          index_pair index_range_at(index x, index y) const noexcept {
              auto it = index_range.find({x, y});
              return (it != index_range.end()) ? it->second : index_pair{0, 0};
          }
      
          void notify(index x, index y) noexcept {
              if (curr_grade != index_pair{x, y}) {
                  grades.push({x, y});
              }
          }
      
      
      private:
          template<typename GradedMatrix>
          void initialize(GradedMatrix& M) {
              index_pair last_pair = {std::numeric_limits<index>::min(), std::numeric_limits<index>::min()};
              index curr_start = std::numeric_limits<index>::min();
      
              for (index i = 0; i < M.get_num_cols(); ++i) {
                  index curr_x = M.z2_col_degrees[i].first;
                  index curr_y = M.z2_col_degrees[i].second;
      
                  assert(curr_x < static_cast<index>(M.x_grid.size()));
                  assert(curr_y < static_cast<index>(M.y_grid.size()));
      
                  if (curr_x != last_pair.first || curr_y != last_pair.second) {
                      // New grade encountered
                      if (curr_start != std::numeric_limits<index>::min()) {
                          index_range[last_pair] = {curr_start, i};
                      }
      
                      curr_start = i;
                      last_pair = {curr_x, curr_y};
                      grades.push(last_pair);
                  }
              }
      
              // Store the final range
              if (curr_start != std::numeric_limits<index>::min()) {
                  index_range[last_pair] = {curr_start, M.get_num_cols()};
              }
          }
  };
  
  } // namespace graded_linalg
  
  #endif // GRID_SCHEDULER_HPP