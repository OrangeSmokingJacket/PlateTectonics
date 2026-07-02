set(LIB_NAME texture_asset)

set(HEADERS
        colors.hpp
        texel_data.hpp
        texture.hpp
)

set(SOURCES
        colors.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PUBLIC
        math
        base_asset
)
