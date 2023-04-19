#pragma once

#include "qrw/Params.hpp"

namespace qrw {

class IOCPAbstract {
 public:
  IOCPAbstract(Params const& params);
  virtual ~IOCPAbstract() = default;

  virtual void solve(std::size_t k) = 0;

  Params params_;
};

}  // namespace qrw
