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

#include<mpp_utils/basic.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace multi_critical {

    typedef mpp_utils::index index;

#if __has_include("ext_interface/backend_log_flag.hpp")
    inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::multi_critical> verbose;
    inline multipers::backend_log_policy::constant_flag<false> very_verbose;
#else
    bool verbose = false;
    bool very_verbose=false;
#endif

} // of namespace multi-critical
