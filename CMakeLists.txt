cmake_minimum_required (VERSION 2.6.0)
project(AtlasSphere)
set (CMAKE_CXX_STANDARD 11)
set(PYBIND11_CPP_STANDARD -std=gnu++11)

find_package(ompl 1.4.2 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OMPL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
link_directories(${OMPL_LIBRARY_DIRS})

add_subdirectory(pybind11)
add_subdirectory(src bin)