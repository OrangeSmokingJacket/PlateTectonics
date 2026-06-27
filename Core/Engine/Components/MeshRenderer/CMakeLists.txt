set (LIB_NAME mesh_renderer_component)

set(HEADERS
	mesh_renderer.hpp
)

set(SOURCES
	mesh_renderer.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
	base_component
	mesh_asset
	material_asset
)