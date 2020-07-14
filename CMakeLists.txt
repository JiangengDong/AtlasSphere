cmake_minimum_required (VERSION 3.10)
project(AtlasSphere)

find_package(Torch QUIET REQUIRED)
if(TARGET torch)
    message(STATUS "Found target torch")
endif(TARGET torch)

find_package(ompl 1.4.2 REQUIRED)
add_library(ompl::ompl INTERFACE IMPORTED)
target_link_libraries(ompl::ompl 
    INTERFACE ${OMPL_LIBRARIES})
target_include_directories(ompl::ompl 
    INTERFACE ${OMPL_INCLUDE_DIRS})
message(STATUS "Found target ompl::ompl")

find_package(Eigen3 REQUIRED)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen 
    PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIRS})
message(STATUS "Found target Eigen3::Eigen")

find_package(Boost 1.58 QUIET REQUIRED COMPONENTS program_options filesystem)
if(TARGET Boost::headers AND TARGET Boost::program_options AND TARGET Boost::filesystem)
    message(STATUS "Found target Boost::headers, Boost::program_options and Boost::filesystem")
endif(TARGET Boost::headers AND TARGET Boost::program_options AND TARGET Boost::filesystem)

find_package(HDF5 QUIET REQUIRED)

add_subdirectory(third-party/yaml-cpp)
if(TARGET yaml-cpp::yaml-cpp)
    message(STATUS "Found target yaml-cpp::yaml-cpp")
endif(TARGET yaml-cpp::yaml-cpp)

add_subdirectory(src)