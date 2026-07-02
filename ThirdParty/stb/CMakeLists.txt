set(LIB_NAME stb)

set(HEADERS
        stb_image.h
)

set(SOURCES
        stb_image.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ../..)
