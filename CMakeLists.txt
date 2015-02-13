cmake_minimum_required(VERSION 2.8.4)
project(EIGEN_CHOL)

# find armadillo
# N.B. FindArmadillo can't seem to figure out how to parse the stupid spaces/parentheses in the windows program files folder, so use an env var instead
#find_package(Armadillo REQUIRED)
set(ARMADILLO_INCLUDE_DIRS "$ENV{ARMADILLO_ROOT}/include")
set(ARMADILLO_LIBRARIES "$ENV{ARMADILLO_ROOT}/lib/armadillo.dll.a")
message(STATUS "Armadillo include dir: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "Armadillo lib: ${ARMADILLO_LIBRARIES}")

set(OPENBLAS_LIBRARIES $ENV{OPENBLAS_ROOT}/lib/libopenblas.dll.a)
message(STATUS "OpenBLAS lib: ${OPENBLAS_LIBRARIES}")

set(BOOST_MIN_VERSION "1.53.0")
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS chrono system REQUIRED)
message(STATUS "Boost lib dir: ${Boost_LIBRARY_DIR}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIR}")

# find Eigen
set(EIGEN_INCLUDE_DIRS $ENV{EIGEN_ROOT})
message(STATUS "Eigen include dir: ${EIGEN_INCLUDE_DIRS}")

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  message(WARN "Building in DEBUG mode will not produce representative results.")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

link_directories(${Boost_LIBRARY_DIR})
include_directories(SYSTEM ${Boost_INCLUDE_DIR} ${ARMADILLO_INCLUDE_DIRS} ${OPENBLAS_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})

add_executable(eigen-chol eigen-chol.cpp)
target_link_libraries(eigen-chol ${Boost_LIBRARIES} ${ARMADILLO_LIBRARIES} ${OPENBLAS_LIBRARIES} gfortran pthread)
