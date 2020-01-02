cc_library(tensor SRCS tensor.cc)
cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)