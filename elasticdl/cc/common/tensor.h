#ifndef ELASTICDL_CC_COMMON_TENSOR_H_
#define ELASTICDL_CC_COMMON_TENSOR_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "elasticdl/cc/common/type.h"

namespace elasticdl {
namespace common {

class Tensor {
 public:
  Tensor(const std::string& name,
         const std::vector<int64_t>& dim,
         const ElemType& type,
         const std::vector<int64_t>& indices = std::vector<int64_t>())
      : name_(name), dim_(dim), element_type_(type), indices_(indices) {
    int64_t size = GetSize() * GetElementSize(element_type_);
    data_ = new char[size];
  }
  Tensor(const std::string& name,
         const std::vector<int64_t>& dim,
         const ElemType& type,
         void* data,
         const std::vector<int64_t>& indices = std::vector<int64_t>())
      : name_(name),
        dim_(dim),
        element_type_(type),
        data_(reinterpret_cast<char*>(data)),
        indices_(indices) {
    is_unowned_ = true;
  }

  ~Tensor() {
    if (!is_unowned()) {
      delete[] data_;
    }
  }

  bool is_unowned() { return is_unowned_; }

  template <class T>
  T* GetRawDataPointer() {
    assert(IsType<T>(element_type_));
    return reinterpret_cast<T*>(data_);
  }

  std::vector<int64_t>& indices() { return indices_; }

  std::vector<int64_t>& dim() { return dim_; }

  std::string& name() { return name_; }

  int64_t GetHeight();

  int64_t GetWidth();

  int64_t GetSize();

  template <class T>
  T& at(size_t index) {
    auto* data = GetRawDataPointer<T>();
    return data[index];
  }

 private:
  std::string name_;
  std::vector<int64_t> dim_;
  ElemType element_type_;
  char* data_{nullptr};
  std::vector<int64_t> indices_;
  bool is_unowned_{false};
};

}  // namespace common
}  // namespace elasticdl

#endif