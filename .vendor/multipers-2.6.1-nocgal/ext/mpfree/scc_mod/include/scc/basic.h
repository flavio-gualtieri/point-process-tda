#pragma once

#include<phat/helpers/misc.h>

namespace scc {

  typedef phat::index index;

  bool verbose = false;

  class ParseError {
  public:
    ParseError(std::string str="") {
      std::cerr << str << std::endl;
    }
  };


} // of namespace scc
