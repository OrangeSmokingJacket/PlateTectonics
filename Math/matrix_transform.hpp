#pragma once

#include "impl/matrix.hpp"
#include "quaternion_ops.hpp"

namespace math::transform {

    template <typename T>
    concept PassableByValue = requires { std::is_trivially_copyable_v<T> && sizeof(T) <= 16; };

    template <typename T>
    constexpr impl::matrix<T, 4, 4> translation_matrix(const impl::vector<T, 3>& vec) noexcept {
        auto result = impl::matrix<T, 4, 4>::identity();
        result[3][0] = vec[0];
        result[3][1] = vec[1];
        result[3][2] = vec[2];
        return result;
    }

    template <typename T>
    constexpr impl::matrix<T, 4, 4> scaling_matrix(const impl::vector<T, 3>& vec) noexcept {
        auto result = impl::matrix<T, 4, 4>::identity();
        result[0][0] = vec[0];
        result[1][1] = vec[1];
        result[2][2] = vec[2];
        return result;
    }

    template <PassableByValue T>
    constexpr impl::matrix<T, 4, 4> rotation_matrix_x(T radians) noexcept {
        T cos = std::cos(radians);
        T sin = std::sin(radians);
        return impl::matrix<T, 4, 4>{impl::vector<T, 4>{T{1}, T{0}, T{0}, T{0}},
                                     impl::vector<T, 4>{T{0}, cos, sin, T{0}},
                                     impl::vector<T, 4>{T{0}, -sin, cos, T{0}},
                                     impl::vector<T, 4>{T{0}, T{0}, T{0}, T{1}}};
    }

    template <PassableByValue T>
    constexpr impl::matrix<T, 4, 4> rotation_matrix_y(T radians) noexcept {
        T cos = std::cos(radians);
        T sin = std::sin(radians);
        return impl::matrix<T, 4, 4>{impl::vector<T, 4>{cos, T{0}, -sin, T{0}},
                                     impl::vector<T, 4>{T{}, T{1}, T{0}, T{0}},
                                     impl::vector<T, 4>{sin, T{0}, cos, T{0}},
                                     impl::vector<T, 4>{T{0}, T{0}, T{0}, T{1}}};
    }

    template <PassableByValue T>
    constexpr impl::matrix<T, 4, 4> rotation_matrix_z(T radians) noexcept {
        T cos = std::cos(radians);
        T sin = std::sin(radians);
        return impl::matrix<T, 4, 4>{impl::vector<T, 4>{cos, sin, T{0}, T{0}},
                                     impl::vector<T, 4>{-sin, cos, T{0}, T{0}},
                                     impl::vector<T, 4>{T{0}, T{0}, T{1}, T{0}},
                                     impl::vector<T, 4>{T{0}, T{0}, T{0}, T{1}}};
    }

    // when cos and sin are already known
    // assumes that axis is normalized
    template <PassableByValue T>
    constexpr impl::matrix<T, 4, 4> rotation_matrix(const impl::vector<T, 3>& axis, T cos, T sin) noexcept {
        T one_minus_cos = T{1} - cos;
        return impl::matrix<T, 4, 4>{impl::vector<T, 4>{axis[0] * axis[0] * one_minus_cos + cos,
                                                        axis[0] * axis[1] * one_minus_cos + axis[2] * sin,
                                                        axis[0] * axis[2] * one_minus_cos - axis[1] * sin,
                                                        T{0}},
                                     impl::vector<T, 4>{axis[1] * axis[0] * one_minus_cos - axis[2] * sin,
                                                        axis[1] * axis[1] * one_minus_cos + cos,
                                                        axis[1] * axis[2] * one_minus_cos + axis[0] * sin,
                                                        T{0}},
                                     impl::vector<T, 4>{axis[2] * axis[0] * one_minus_cos + axis[1] * sin,
                                                        axis[2] * axis[1] * one_minus_cos - axis[0] * sin,
                                                        axis[2] * axis[2] * one_minus_cos + cos,
                                                        T{0}},
                                     impl::vector<T, 4>{T{0}, T{0}, T{0}, T{1}}};
    }

    // assumes that axis is normalized
    template <PassableByValue T>
    constexpr impl::matrix<T, 4, 4> rotation_matrix(const impl::vector<T, 3>& axis, T radians) noexcept {
        return rotation_matrix(axis, std::cos(radians), std::sin(radians));
    }

    template <PassableByValue T>
    constexpr impl::matrix<T, 4, 4> rotation_matrix(const impl::quaternion<T>& q) noexcept {
        return impl::matrix<T, 4, 4>{
            impl::vector<T, 4>{T{1} - T{2} * (q.y * q.y + q.z * q.z),
                               T{2} * (q.x * q.y + q.w * q.z),
                               T{2} * (q.x * q.z - q.w * q.y),
                               T{0}},
            impl::vector<T, 4>{T{2} * (q.x * q.y - q.w * q.z),
                               T{1} - T{2} * (q.x * q.x + q.z * q.z),
                               T{2} * (q.y * q.z + q.w * q.x),
                               T{0}},
            impl::vector<T, 4>{T{2} * (q.x * q.y + q.w * q.z),
                               T{2} * (q.y * q.z - q.w * q.x),
                               T{1} - T{2} * (q.x * q.x + q.y * q.y),
                               T{0}},
            impl::vector<T, 4>{T{0}, T{0}, T{0}, T{1}},
        };
    }

    // TODO: orthographic
    template <PassableByValue T>
    constexpr impl::matrix<T, 4, 4> perspective_matrix(T fov_radians, T aspect_ratio, T near, T far) noexcept {
        T scaling = 1 / std::tan(fov_radians / T{2});
        return impl::matrix<T, 4, 4>{impl::vector<T, 4>{scaling / aspect_ratio, T{0}, T{0}, T{0}},
                                     impl::vector<T, 4>{T{0}, scaling, T{0}, T{0}},
                                     impl::vector<T, 4>{T{0}, T{0}, (far + near) / (near - far), T{-1}},
                                     impl::vector<T, 4>{T{0}, T{0}, far * near * T{2} / (near - far), T{0}}};
    }

} // namespace math::transform
