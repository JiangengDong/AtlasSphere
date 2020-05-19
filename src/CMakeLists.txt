include_directories(./include)

add_library(SpherePlanning MODULE
        python_plugin.cpp
        SpherePlanning.cpp
        RRTConnect.cpp
        RRTstar.cpp
        AtlasStateSpace.cpp)
target_include_directories(SpherePlanning PRIVATE ${OMPL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(SpherePlanning PRIVATE ${OMPL_LIBRARIES} pybind11::module)
set_target_properties(SpherePlanning PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                SUFFIX "${PYTHON_MODULE_EXTENSION}"
                                                LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts"
                                                CXX_STANDARD 14)