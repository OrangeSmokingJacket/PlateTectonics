cmake_minimum_required(VERSION 3.25)
project(PlateTectonics LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

option(ENABLE_ASAN "use address sanitizer" OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()


if (ENABLE_ASAN)
    if (MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>)
        add_link_options($<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>)
    else ()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
        add_link_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    endif()
    message("Address sanitizer is enabled")
else ()
    message("Address sanitizer is disabled")
endif()

include(FetchContent)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCilab/cereal.git
        GIT_TAG v1.3.2
)
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        docking
)
# TODO: remove
FetchContent_Declare(
        glm
        GIT_REPOSITORY	https://github.com/g-truc/glm.git
        GIT_TAG 1.0
)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

set(JUST_INSTALL_CEREAL ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw)
FetchContent_MakeAvailable(cereal)
FetchContent_MakeAvailable(imgui)
# TODO: remove
FetchContent_MakeAvailable(glm)

add_compile_definitions(SHADERS_DIR="${PROJECT_SOURCE_DIR}/Shaders")
add_compile_definitions(ASSETS_DIR="${PROJECT_SOURCE_DIR}/Assets")

add_subdirectory(ThirdParty)
add_subdirectory(Math)
add_subdirectory(Core)
add_subdirectory(Utils)

add_executable(${PROJECT_NAME}
        main.cpp

        App/app_layer.cpp
        App/simulation_domain.cpp
        App/euler_fluid_sim.cpp
        App/simulation_kernel.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        imgui_lib
        application
        input_system
        renderer
        scene
        asset_manager
        asset_loader
        cereal::cereal

        CUDA::cudart

        camera_component
        mesh_renderer_component
        scriptable_component
)

# TODO: warnings only for this project
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Wall>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->

        $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>
        $<$<COMPILE_LANGUAGE:CXX>:/external:W0>
        $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>
        $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>

        $<$<COMPILE_LANGUAGE:CXX>:-fconcepts-diagnostics-depth=10>
        $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    )
endif()
