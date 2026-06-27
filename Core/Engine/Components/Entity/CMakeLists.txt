set (LIB_NAME entity)

set(HEADERS
	entity.hpp
)

set(SOURCES
	entity.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
)