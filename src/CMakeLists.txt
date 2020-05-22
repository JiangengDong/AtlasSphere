include_directories(./include)

add_executable(SpherePlanningExecutable
        main.cpp
        SpherePlanning.cpp
        RRTConnect.cpp
        RRTstar.cpp
        # AtlasStateSpace.cpp
        MPNetPlanner.cpp
        MPNetSampler.cpp)
target_include_directories(SpherePlanningExecutable PRIVATE ${OMPL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(SpherePlanningExecutable PRIVATE ${OMPL_LIBRARIES} ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(SpherePlanningExecutable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
                                                CXX_STANDARD 11
                                                COMPILE_FLAGS "--std=gnu++11")