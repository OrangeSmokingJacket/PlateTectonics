set(LIB_NAME asset_manager)

set(HEADERS
	asset_manager.hpp
)

set(SOURCES
	asset_manager.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_BINARY_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
	base_asset
    asset_loader
)
