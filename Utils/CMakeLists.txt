set(LIB_NAME utils)

set(HEADERS
    array_utils.hpp
	buffer_utils.hpp
    compiler_utils.hpp
	enum_utils.hpp
    thread_pool.hpp
	hash.hpp
    uuid.hpp
)

set(SOURCES
    hash.cpp
    uuid.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
        cereal::cereal
)
