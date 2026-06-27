set(LIB_NAME renderer)

set(HEADERS
        buffer_objects.hpp
        shader_generator.hpp
        shader_program.hpp
        variant_shader_program.hpp
        renderer.hpp
)

set(SOURCES
        buffer_objects.cpp
        shader_generator.cpp
        shader_program.cpp
        variant_shader_program.cpp
        renderer.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(
        ${LIB_NAME} PUBLIC
        glad
        glfw
        OpenGL::GL
        utils
        entity
        filesystem
)