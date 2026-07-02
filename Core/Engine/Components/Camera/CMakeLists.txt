set (LIB_NAME camera_component)

set(HEADERS
	camera.hpp
)

set(SOURCES
	camera.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
	base_component
    math
)
