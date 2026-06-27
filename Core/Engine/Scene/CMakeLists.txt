set(LIB_NAME scene)

set(HEADERS
        scene.hpp
)

set(SOURCES
        scene.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(
        ${LIB_NAME} PUBLIC
        transform_component
        entity
        entity_manager
)