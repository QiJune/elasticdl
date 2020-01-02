cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(ELASTICDL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ELASTICDL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/bazel.cmake/cmake)

project(elascticdl CXX C)

set(WITH_GPU OFF)
include(bazel)
add_subdirectory(elasticdl)
