cmake_minimum_required (VERSION 3.10)
project(AtlasSphere)

find_package(ompl 1.4.2 REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(pybind11)
add_subdirectory(src)