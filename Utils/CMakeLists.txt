set(LIB_NAME utils)

set(HEADERS
    array_utils.hpp
	buffer_utils.hpp
    compiler_utils.hpp
    cuda_def.hpp
	enum_utils.hpp
    thread_pool.hpp
	hash.hpp
    template_utils.hpp
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
        CUDA::cudart
)

if (MSVC)
    target_compile_options(${LIB_NAME} PUBLIC
            # required for cuda::std
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()
