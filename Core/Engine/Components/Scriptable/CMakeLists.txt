set(LIB_NAME scriptable_component)

set(HEADERS
    scriptable.hpp
)

set(SOURCES
    scriptable.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
    base_component
)
