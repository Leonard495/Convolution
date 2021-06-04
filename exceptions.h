#pragma once

#include <sstream>
#include <stdexcept>

#define THROW(text)				\
if(true){					\
  std::stringstream str;			\
  str << text;					\
  throw std::runtime_error(str.str());	\
};
