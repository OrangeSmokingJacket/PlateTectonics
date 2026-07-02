set(LIB_NAME mesh_asset)

set(HEADERS
    mesh.hpp
)

set(SOURCES
    mesh.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(
        ${LIB_NAME} PUBLIC
        base_asset
        cereal::cereal
)
