#pragma once

#include "matrix_ops.hpp"
#include "quaternion_ops.hpp"
#include "scalar_ops.hpp"
#include "vector_ops.hpp"

namespace math {

    template <typename T>
    using vec2 = impl::vector<T, 2>;

    template <typename T>
    using vec3 = impl::vector<T, 3>;

    template <typename T>
    using vec4 = impl::vector<T, 4>;

    template <typename T>
    using mat2x2 = impl::matrix<T, 2, 2>;

    template <typename T>
    using mat2x3 = impl::matrix<T, 2, 3>;

    template <typename T>
    using mat2x4 = impl::matrix<T, 2, 4>;

    template <typename T>
    using mat3x2 = impl::matrix<T, 3, 2>;

    template <typename T>
    using mat3x3 = impl::matrix<T, 3, 3>;

    template <typename T>
    using mat3x4 = impl::matrix<T, 3, 4>;

    template <typename T>
    using mat4x2 = impl::matrix<T, 3, 2>;

    template <typename T>
    using mat4x3 = impl::matrix<T, 3, 3>;

    template <typename T>
    using mat4x4 = impl::matrix<T, 4, 4>;

    template <typename T>
    using quaternion = impl::quaternion<T>;

} // namespace math
