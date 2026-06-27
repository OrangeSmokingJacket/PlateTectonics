set(LIB_NAME glad)

set(HEADERS
        glad.h
        KHR/khrplatform.h
)

set(SOURCES
        glad.c
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ../..)
