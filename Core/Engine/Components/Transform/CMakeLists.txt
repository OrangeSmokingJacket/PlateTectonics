set(LIB_NAME transform_component)

set(HEADERS
    transform.hpp
)

set(SOURCES
    transform.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(
        ${LIB_NAME} PUBLIC
        math
)