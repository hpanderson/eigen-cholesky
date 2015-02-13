cmake_minimum_required(VERSION 2.8.4)
project(EIGEN_CHOL)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  message(WARN "Building in DEBUG mode will not produce representative results.")
endif ()

add_executable(eigen-chol eigen-chol.cpp)
