add_executable(SpherePlanningExecutable
        main.cpp
        SpherePlanning.cpp
        RRTConnect.cpp
        RRTstar.cpp
        MPNetPlanner.cpp
        MPNetSampler.cpp
        Parameter.cpp)
target_include_directories(SpherePlanningExecutable
    PRIVATE
        ./include)
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
