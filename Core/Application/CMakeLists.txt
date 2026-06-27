set(LIB_NAME application)

set(HEADERS
        application.hpp
        layer.hpp
        window.hpp
)

set(SOURCES
        application.cpp
        window.cpp
)

add_library(${LIB_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} PRIVATE
        glad
        glfw
        OpenGL::GL
        imgui_lib
        glm::glm
)
