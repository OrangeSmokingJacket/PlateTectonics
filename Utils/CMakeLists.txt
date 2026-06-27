set(LIB_NAME utils)

set(HEADERS
	buffer_utils.hpp
	enum_utils.hpp
	hash.hpp
)

set(SOURCES
    uuid.cpp
    hash.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
        cereal::cereal
)
