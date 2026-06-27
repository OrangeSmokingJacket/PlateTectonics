set(LIB_NAME filesystem)

set(HEADERS
	filesystem.hpp
)

set(SOURCES
	filesystem.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
	utils
)