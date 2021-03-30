add_library(SpherePlanning
    ./spherePlanning/src/SpherePlanning.cpp
    ./spherePlanning/src/Parameter.cpp
    ./spherePlanning/planner/src/RRTConnect.cpp
    )
target_include_directories(SpherePlanning
    PUBLIC
        ./spherePlanning)
target_link_libraries(SpherePlanning
    PUBLIC
        Eigen3::Eigen
        ompl_interface
        # torch
        Boost::boost
        Boost::program_options
        Boost::filesystem)

add_executable(GenerateTrainingStartAndGoal
    ./executable/GenerateTrainingStartAndGoal.cpp)
target_link_libraries(GenerateTrainingStartAndGoal
    PUBLIC SpherePlanning cnpy-static)

add_executable(GenerateTrainingPath
    ./executable/GenerateTrainingPath.cpp)
target_link_libraries(GenerateTrainingPath
    PUBLIC SpherePlanning cnpy-static)

add_executable(GeneratePointCloud
    ./executable/GeneratePointCloud.cpp)
target_link_libraries(GeneratePointCloud
    PUBLIC SpherePlanning cnpy-static)
