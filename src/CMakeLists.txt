add_executable(SpherePlanningExecutable
        main.cpp
        SpherePlanning.cpp
        Parameter.cpp
        planner/RRTConnect.cpp
        planner/RRTstar.cpp
        planner/MPNetPlanner.cpp
        planner/MPNetSampler.cpp
        planner/FMT.cpp 
        )
target_include_directories(SpherePlanningExecutable
    PRIVATE
        ./include
        ./planner/include)
target_link_libraries(SpherePlanningExecutable
    PRIVATE
        Eigen3::Eigen
        ompl::ompl
        torch
        Boost::headers
        Boost::program_options
        Boost::filesystem)
set_target_properties(SpherePlanningExecutable
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
        CXX_STANDARD 11)
