set(LIB_NAME entity_manager)

set(HEADERS
    entity_manager.hpp
    basic_entities.hpp
)

set(SOURCES
    entity_manager.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
    entity
    # for basic entities
    mesh_asset
    material_asset
    transform_component
)
