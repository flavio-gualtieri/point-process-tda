/* Copyright 2021 TU Graz
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

#include<mpp_utils/basic.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace multi_chunk {

  typedef mpp_utils::index index;

#if __has_include("ext_interface/backend_log_flag.hpp")
  inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::multi_critical> verbose;
#else
  bool verbose = true;
#endif
}

