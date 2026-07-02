#pragma once

#include "vector.hpp"

namespace math::impl {

    // W first, vector part after
    template <typename T>
    class quaternion {
    public:
        // Identity
        explicit constexpr quaternion();
        // Pure quaternion
        explicit constexpr quaternion(const vector<T, 3>& vector) noexcept;
        explicit constexpr quaternion(T w, const vector<T, 3>& vector_part) noexcept;

        [[nodiscard]] constexpr quaternion conjugate() const noexcept;
        [[nodiscard]] quaternion inverse() const noexcept;

        [[nodiscard]] T magnitude() const noexcept;

        quaternion& normalize() noexcept;
        quaternion& normalize_safe() noexcept;

        [[nodiscard]] constexpr bool is_pure() const noexcept;

        static quaternion from_axis_angle(T angle, const vector<T, 3>& axis);

        T w;
        union {
            impl::vector<T, 3> vector;
            struct {
                T x, y, z;
            };
        };
    };

    // Constructors

    template <typename T>
    constexpr quaternion<T>::quaternion()
        : w(T{1})
        , x(T{0})
        , y(T{0})
        , z(T{0}) {}

    template <typename T>
    constexpr quaternion<T>::quaternion(const impl::vector<T, 3>& vector) noexcept
        : w(T{0})
        , vector(vector) {}

    // Methods

    template <typename T>
    constexpr quaternion<T> quaternion<T>::conjugate() const noexcept {
        quaternion result;
        result.w = w;
        result.vector = -vector;
        return result;
    }

    template <typename T>
    constexpr quaternion<T>::quaternion(T w, const impl::vector<T, 3>& v) noexcept
        : w(w)
        , vector(v) {}

    template <typename T>
    quaternion<T> quaternion<T>::inverse() const noexcept {
        return conjugate() / magnitude();
    }

    template <typename T>
    T quaternion<T>::magnitude() const noexcept {
        return std::sqrt(vector.magnitude_sqr() + w * w);
    }

    template <typename T>
    quaternion<T>& quaternion<T>::normalize() noexcept {
        auto mag = magnitude();
        assert(mag != T{0} && "division by zero in unsafe quaternion::normalize()");
        w /= mag;
        vector /= mag;
        return *this;
    }

    template <typename T>
    quaternion<T>& quaternion<T>::normalize_safe() noexcept {
        auto mag = magnitude();
        if (mag != T{0}) {
            w /= mag;
            vector /= mag;
        }
        return *this;
    }

    template <typename T>
    quaternion<T> quaternion<T>::from_axis_angle(T angle, const impl::vector<T, 3>& axis) {
        auto cos = std::cos(angle / 2);
        auto sin = std::sin(angle / 2);
        return quaternion(cos, sin * axis);
    }

    template <typename T>
    constexpr bool quaternion<T>::is_pure() const noexcept {
        return w == T{0};
    }

    // Arithmetics

    template <typename T>
    constexpr quaternion<T> operator+(const quaternion<T>& lhs, const quaternion<T>& rhs) noexcept {
        quaternion<T> result;
        result.w = lhs.w + rhs.w;
        result.vector = lhs.vector + rhs.vector;
        return result;
    }

    template <typename T>
    constexpr quaternion<T>& operator+=(quaternion<T>& lhs, const quaternion<T>& rhs) noexcept {
        lhs.w += rhs.w;
        lhs.vector += rhs.vector;
        return lhs;
    }

    template <typename T>
    constexpr quaternion<T> operator-(const quaternion<T>& lhs, const quaternion<T>& rhs) noexcept {
        quaternion<T> result;
        result.w = lhs.w - rhs.w;
        result.vector = lhs.vector - rhs.vector;
        return result;
    }

    template <typename T>
    constexpr quaternion<T>& operator-=(quaternion<T>& lhs, const quaternion<T>& rhs) noexcept {
        lhs.w -= rhs.w;
        lhs.vector -= rhs.vector;
        return lhs;
    }

    template <typename T>
    constexpr quaternion<T> operator*(const quaternion<T>& lhs, T val) noexcept {
        quaternion<T> result;
        result.w = lhs.w * val;
        result.vector = lhs.vector * val;
        return result;
    }

    template <typename T>
    constexpr quaternion<T> operator*(const quaternion<T>& lhs, const quaternion<T>& rhs) noexcept {
        // TODO: profile which one is faster
        /*
        quaternion<T> result;
        result.w = (lhs.w * rhs.w) - (lhs.x * rhs.x) - (lhs.y * rhs.y) - (lhs.z * rhs.z);
        result.x = (lhs.w * rhs.x) + (lhs.x * rhs.w) + (lhs.y * rhs.z) - (lhs.z * rhs.y);
        result.y = (lhs.w * rhs.y) - (lhs.x * rhs.z) + (lhs.y * rhs.w) + (lhs.z * rhs.x);
        result.z = (lhs.w * rhs.z) + (lhs.x * rhs.y) - (lhs.y * rhs.x) + (lhs.z * rhs.w);
        return result;
        */
        return quaternion<T>{lhs.w * rhs.w - dot_product(lhs.vector, rhs.vector),
                             lhs.w * rhs.vector + rhs.w * lhs.vector + cross_product(lhs.vector, rhs.vector)};
    }

    template <typename T>
    constexpr quaternion<T>& operator*=(quaternion<T>& lhs, const quaternion<T>& rhs) noexcept {
        // TODO: vector vs component based computation
        quaternion<T> result;
        result.w = (lhs.w * rhs.w) - (lhs.x * rhs.x) - (lhs.y * rhs.y) - (lhs.z * rhs.z);
        result.x = (lhs.w * rhs.x) + (lhs.x * rhs.w) + (lhs.y * rhs.z) - (lhs.z * rhs.y);
        result.y = (lhs.w * rhs.y) - (lhs.x * rhs.z) + (lhs.y * rhs.w) + (lhs.z * rhs.x);
        result.z = (lhs.w * rhs.z) + (lhs.x * rhs.y) - (lhs.y * rhs.x) + (lhs.z * rhs.w);
        lhs = result;
        return lhs;
    }

    template <typename T>
    constexpr quaternion<T>& operator*=(quaternion<T>& lhs, T val) noexcept {
        lhs.w *= val;
        lhs.vector *= val;
        return lhs;
    }

    template <typename T>
    constexpr quaternion<T> operator/(const quaternion<T>& lhs, T val) noexcept {
        quaternion<T> result;
        result.w = lhs.w / val;
        result.vector = lhs.vector / val;
        return result;
    }

    template <typename T>
    constexpr quaternion<T>& operator/=(quaternion<T>& lhs, T val) noexcept {
        lhs.w /= val;
        lhs.vector /= val;
        return lhs;
    }

} // namespace math::impl
