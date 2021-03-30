cmake_minimum_required (VERSION 3.10)
project(AtlasSphere)

# torch
find_package(Torch QUIET REQUIRED)
if(TARGET torch)
    message(STATUS "Found target torch")
endif(TARGET torch)

# ompl: wrap into an interface library
find_package(ompl 1.4.2 REQUIRED)
add_library(ompl_interface INTERFACE)
target_link_libraries(ompl_interface
    INTERFACE ${OMPL_LIBRARIES})
target_include_directories(ompl_interface
    INTERFACE ${OMPL_INCLUDE_DIRS})
message(STATUS "Found target ompl::ompl")

# eigen3
find_package(Eigen3 REQUIRED)

# boost
find_package(Boost 1.58 QUIET REQUIRED COMPONENTS program_options filesystem)
if(TARGET Boost::headers AND TARGET Boost::program_options AND TARGET Boost::filesystem)
    message(STATUS "Found target Boost::headers, Boost::program_options and Boost::filesystem")
endif(TARGET Boost::headers AND TARGET Boost::program_options AND TARGET Boost::filesystem)

# cnpy: its original CMakeLists.txt is not well organized, so rewrite its target here
find_package(ZLIB)
add_library(cnpy-static STATIC
    extern/cnpy/cnpy.cpp)
target_link_libraries(cnpy-static PUBLIC
    ${ZLIB_LIBRARIES})
target_include_directories(cnpy-static PUBLIC
    ${ZLIB_INCLUDE_DIRS}
    ./extern)

add_subdirectory(src)
