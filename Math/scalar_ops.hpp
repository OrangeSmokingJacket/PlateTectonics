#pragma once

#include "constants.hpp"

#include <random>

namespace math {

    template<typename T>
    bool constexpr is_equal(T x, T y) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::fabs(x - y) < std::numeric_limits<T>::epsilon();
        } else {
            return x == y;
        }
    }

    template<typename T>
    T square(T val) { return val * val; }

    template<typename T>
    T cube(T val) { return val * val * val; }

	template<typename T>
	T to_radians(T val) {
		return val * constants::PI<T> / T{ 180 };
	}

	template<typename T>
	T to_degrees(T val) {
		return val * T{ 180 } / constants::PI<T>;
	}

    template<typename T> requires (std::is_floating_point_v<T>)
    T random() {
        thread_local std::random_device rd;
	    thread_local std::mt19937 engine(rd());
	    thread_local std::uniform_real_distribution<T> distribution(T{0.0}, T{1.0});

	    return distribution(engine);
	}

    template <typename T>
    T lerp(T a, T b, float t) {
	    return a * (1 - t) + b * t;
	}

}