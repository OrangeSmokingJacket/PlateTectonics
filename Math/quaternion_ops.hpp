#pragma once

#include "impl/quaternion.hpp"

namespace math {

    template<typename T>
    impl::vector<T, 3> apply_rotation(const impl::vector<T, 3>& vector, const impl::quaternion<T>& quaternion) {
        return (quaternion * impl::quaternion<T>(vector) * quaternion.inverse()).vector;
    }

    template<typename T>
    impl::vector<T, 3> rotate(const impl::vector<T, 3>& vector, const impl::vector<T, 3>& axis, T radians) {
        auto cos = std::cos(radians / T{2});
        auto sin = std::sin(radians / T{2});
        return apply_rotation(vector, impl::quaternion<T>(cos, axis * sin));
    }
    // value is in range [0,1]
    template<typename T>
    impl::quaternion<T> slerp(const impl::quaternion<T>& a, const impl::quaternion<T>& b, T value) {
        auto difference = b * a.inverse();
        auto angle = std::acos(difference.w) * T{2};
        auto new_angle = angle * value;
        auto cos = std::cos(new_angle);
        auto sin = std::sin(new_angle);
        return impl::quaternion<T>(cos, sin * (difference).normalize().vector);
    }



}
