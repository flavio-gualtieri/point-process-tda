#pragma once

#include <stdint.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace scc {

  typedef int64_t index;

#if __has_include("ext_interface/backend_log_flag.hpp")
  inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::multi_critical> verbose;
#else
  bool verbose = false;
#endif

  class ParseError {
  public:
    ParseError(std::string str="") {
      std::cerr << str << std::endl;
    }
  };


} // of namespace scc
