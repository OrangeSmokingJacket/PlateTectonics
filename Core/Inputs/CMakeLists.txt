set(LIB_NAME input_system)

set(HEADERS
    inputs.hpp
)

set(SOURCES
    inputs.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
        glfw
        math
        application
)
