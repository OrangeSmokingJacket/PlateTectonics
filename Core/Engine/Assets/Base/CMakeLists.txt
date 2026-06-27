set(LIB_NAME base_asset)

set(HEADERS
	base.hpp
    default_assets.hpp
)

set(SOURCES
	base.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
	utils
)