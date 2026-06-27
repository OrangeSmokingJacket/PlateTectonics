set(LIB_NAME asset_loader)

set(HEADERS
    asset_loader.hpp
)

set(SOURCES
    asset_loader.cpp
    mesh_loader.cpp
    material_loader.cpp
    texture_loader.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_BINARY_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
	base_asset
	mesh_asset
	material_asset
	texture_asset

    cereal::cereal
    stb
)