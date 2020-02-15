cmake_minimum_required (VERSION 2.6.0)
project(AtlasSphere)
set (CMAKE_CXX_STANDARD 11)

find_package(ompl REQUIRED)

include_directories(${OMPL_INCLUDE_DIRS} /usr/include/)
link_directories(${OMPL_LIBRARY_DIRS})

add_subdirectory(src bin)