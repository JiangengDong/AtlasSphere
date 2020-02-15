include_directories(./include)

add_executable(SpherePlanning
        main.cpp
        SpherePlanning.cpp
        RRTConnect.cpp
        AtlasStateSpace.cpp
        )
target_link_libraries(SpherePlanning 
        ${OMPL_LIBRARIES})