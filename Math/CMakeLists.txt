set(LIB_NAME math)

set(HEADERS
	impl/vector.hpp
    impl/quaternion.hpp
	impl/matrix.hpp
	impl/serialization.hpp

	constants.hpp
	math_def.hpp
	scalar_ops.hpp
	vector_ops.hpp
    quaternion_ops.hpp
	matrix_ops.hpp
	matrix_transform.hpp
)

add_library(${LIB_NAME} INTERFACE ${HEADERS})

include_directories(${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} INTERFACE
    CUDA::cudart
    cereal::cereal
    utils
)

if (MSVC)
    target_compile_options(${LIB_NAME} INTERFACE
        # required for cuda::std
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()
