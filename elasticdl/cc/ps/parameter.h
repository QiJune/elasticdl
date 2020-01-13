#ifndef ELASTICDL_CC_PS_PARAMETER_H_
#define ELASTICDL_CC_PS_PARAMETER_H_

#include "elasticdl/cc/common/embedding_table.h"
#include "elasticdl/cc/common/tensor.h"

namespace elasticdl {
namespace ps {

class Parameter {
 public:
  ~Parameter() {
    for (auto& p : non_embedding_params_) {
      delete p.second;
    }

    for (auto& p : embedding_params_) {
      delete p.second;
    }
  }

  void CreateEmbeddingTable(const std::string& name,
                            const common::ElemType& type,
                            int64_t embedding_dim,
                            const std::string& initializer);

  template <typename T>
  void GetEmbeddingParam(const std::string& name,
                         const std::vector<int64_t>& indices,
                         T* buffer,
                         int64_t size) {
    CHECK(embedding_params_.count(name));
    embedding_params_[name]->GetEmbeddingVectors<T>(indices, buffer, size);
  }

  void CreateNonEmbeddingParam(const std::string& name,
                               const common::ElemType& type,
                               const std::vector<int64_t>& dim,
                               void* data);

  bool is_init() { return is_init_; }

  bool version() { return version_; }

  std::unordered_map<std::string, common::Tensor*>& non_embedding_params() {
    return non_embedding_params_;
  }

  std::unordered_map<std::string, common::EmbeddingTable*>& embedding_params() {
    return embedding_params_;
  }

 private:
  std::unordered_map<std::string, common::Tensor*> non_embedding_params_;
  std::unordered_map<std::string, common::EmbeddingTable*> embedding_params_;
  int64_t version_{-1};
  bool is_init_{false};
};

}  // namespace ps
}  // namespace elasticdl
#endif