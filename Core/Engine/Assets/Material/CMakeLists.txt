set(LIB_NAME material_asset)

set(HEADERS
    material.hpp
)

set(SOURCES
    material.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(
        ${LIB_NAME} PUBLIC
        math
        texture_asset
        base_asset
        asset_ref
)
