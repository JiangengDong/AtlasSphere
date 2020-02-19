include_directories(./include)

add_library(SpherePlanning MODULE 
        python_plugin.cpp
        SpherePlanning.cpp
        RRTConnect.cpp
        AtlasStateSpace.cpp)
target_link_libraries(SpherePlanning PRIVATE ${OMPL_LIBRARIES} pybind11::module)
set_target_properties(SpherePlanning PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                SUFFIX "${PYTHON_MODULE_EXTENSION}"
                                                LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/script")

add_executable(SpherePlanningExecutable
        main.cpp
        SpherePlanning.cpp
        RRTConnect.cpp
        AtlasStateSpace.cpp)
target_link_libraries(SpherePlanningExecutable PRIVATE ${OMPL_LIBRARIES})