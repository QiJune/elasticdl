#ifndef ELASTICDL_CC_PS_OPTIMIZER_H_
#define ELASTICDL_CC_PS_OPTIMIZER_H_

#include "elasticdl/cc/common/tensor.h"
#include "elasticdl/cc/ps/parameter.h"

namespace elasticdl {
namespace ps {

class Optimizer {
 public:
  virtual void ApplyGradients(const std::vector<common::Tensor>& grads,
                              Parameter* p) = 0;

  double lr() { return lr_; }

 private:
  double lr_;
};

class SGDOptimizer : public Optimizer {
 public:
  void ApplyGradients(const std::vector<common::Tensor>& grads, Parameter* p);
};

}  // namespace ps
}  // namespace elasticdl

#endif