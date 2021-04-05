add_library(SpherePlanning
    ./spherePlanning/src/SpherePlanning.cpp
    ./spherePlanning/planner/src/RRTConnect.cpp
    ./spherePlanning/planner/src/MPNetPlanner.cpp
    ./spherePlanning/planner/src/MPNetSampler.cpp
    )
target_include_directories(SpherePlanning
    PUBLIC
        ./spherePlanning)
target_link_libraries(SpherePlanning
    PUBLIC
        Eigen3::Eigen
        ompl_interface
        torch
        Boost::boost
        Boost::program_options
        Boost::filesystem)

add_executable(GenerateBrickConfig
    ./executable/GenerateBrickConfig.cpp)
target_link_libraries(GenerateBrickConfig
    PUBLIC SpherePlanning cnpy-static)

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

add_executable(GenerateTestingStartAndGoal
    ./executable/GenerateTestingStartAndGoal.cpp)
target_link_libraries(GenerateTestingStartAndGoal
    PUBLIC SpherePlanning cnpy-static)

add_executable(TestCoMPNet
    ./executable/TestCoMPNet.cpp)
target_link_libraries(TestCoMPNet
    PUBLIC SpherePlanning cnpy-static)

add_executable(temp
    ./executable/temp.cpp)
target_link_libraries(temp
    PUBLIC SpherePlanning cnpy-static)
