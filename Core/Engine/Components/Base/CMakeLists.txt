set(LIB_NAME base_component)

set(HEADERS
	base.hpp
)

set(SOURCES
	base.cpp
)

add_library(${LIB_NAME} ${HEADER} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
	
)