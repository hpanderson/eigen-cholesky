cmake_minimum_required(VERSION 2.8.4)
project(EIGEN_CHOL)

# find armadillo
# N.B. FindArmadillo can't seem to figure out how to parse the stupid spaces/parentheses in the windows program files folder, so use an env var instead
#find_package(Armadillo REQUIRED)
set(ARMADILLO_INCLUDE_DIRS "$ENV{ARMADILLO_ROOT}/include")
message(STATUS "Armadillo include dir: ${ARMADILLO_INCLUDE_DIRS}")

set(MKL_LIBRARIES "$ENV{INTEL_ROOT}/mkl/lib/intel64/mkl_rt.lib")
message(STATUS "Armadillo lib: ${MKL_LIBRARIES}")

#set(OPENBLAS_LIBRARIES $ENV{OPENBLAS_ROOT}/lib/libopenblas.dll.a)
#message(STATUS "OpenBLAS lib: ${OPENBLAS_LIBRARIES}")

# find Eigen
set(EIGEN_INCLUDE_DIRS $ENV{EIGEN_ROOT})
message(STATUS "Eigen include dir: ${EIGEN_INCLUDE_DIRS}")

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  message(WARN "Building in DEBUG mode will not produce representative results.")
endif ()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -O3")
endif ()

if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_HAS_EXCEPTIONS=0 -U__EXCEPTIONS -U_CPPUNWIND")
endif ()

include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})

add_executable(eigen-chol eigen-chol.cpp)
target_link_libraries(eigen-chol ${OPENBLAS_LIBRARIES} ${MKL_LIBRARIES})
if (NOT WIN32)
  target_link_libraries(eigen-chol gfortran pthread)
endif ()
