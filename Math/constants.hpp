#pragma once

#include <numbers>

namespace math::constants {

    template <typename T>
    auto PI = std::numbers::pi_v<T>;

}