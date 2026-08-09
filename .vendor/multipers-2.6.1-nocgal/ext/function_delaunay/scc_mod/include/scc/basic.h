#pragma once

#include <stdint.h>

namespace scc {

  typedef int64_t index;

  bool verbose = false;

  class ParseError {
  public:
    ParseError(std::string str="") {
      std::cerr << str << std::endl;
    }
  };


} // of namespace scc
