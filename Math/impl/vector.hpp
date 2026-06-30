#pragma once

#include <array>
#include <cmath>
#include <cassert>
#include <cstring>

#include <Utils/hash.hpp>

namespace math::impl {

    template <typename T, size_t Dimensions>
    struct vector_storage {
        std::array<T, Dimensions> arr;
    };
    
    template <typename T>
    struct vector_storage<T, 2> {
        union {
            std::array<T, 2> arr;
            struct { T x, y; };
        };
    };

    template <typename T>
    struct vector_storage<T, 3> {
        union {
            std::array<T, 3> arr;
            struct { T x, y, z; };
            struct { T r, g, b; };
        };
    };

    template <typename T>
    struct vector_storage<T, 4> {
        union {
            std::array<T, 4> arr;
            struct { T x, y, z, w; };
            struct { T r, g, b, a; };
        };
    };

    template <typename T, size_t Dimensions>
    struct vector : vector_storage<T, Dimensions>{

        constexpr vector() noexcept;

        constexpr vector(const vector&) noexcept = default;
        constexpr vector(vector&&) noexcept = default;
        constexpr vector& operator=(const vector&) noexcept = default;
        constexpr vector& operator=(vector&&) noexcept = default;

        constexpr explicit vector(T value);

        template<typename U> requires(std::is_constructible_v<T, U> || std::is_convertible_v<U, T>)
        constexpr explicit vector(U&& value);

        template<size_t FromDimensions> requires (FromDimensions > Dimensions)
        explicit constexpr vector(const vector<T, FromDimensions>& other);

        template<size_t FromDimensions> requires (FromDimensions < Dimensions)
        explicit constexpr vector(const vector<T, FromDimensions>& other);

        template<size_t FromDimensions, typename... Args> requires (FromDimensions < Dimensions && Dimensions - FromDimensions == sizeof...(Args))
        explicit constexpr vector(const vector<T, FromDimensions>& other, Args&&... args);

        template<typename... Args> requires (sizeof...(Args) == Dimensions)
        explicit constexpr vector(Args&&... args) noexcept;

        template<typename U>
        explicit constexpr operator vector<U, Dimensions>() const noexcept;

        consteval static vector zero_vector() noexcept {
            vector result;
            result.arr.fill(T{0});
            return result;
        }

        constexpr T& operator[](size_t i) noexcept;
        constexpr const T& operator[](size_t i) const noexcept;

        constexpr T* data() noexcept;
        constexpr const T* data() const noexcept;

        constexpr size_t size() const noexcept;

        auto begin() noexcept -> std::array<T, Dimensions>::iterator;
        auto end() noexcept -> std::array<T, Dimensions>::iterator;
        auto begin() const noexcept -> std::array<T, Dimensions>::const_iterator;
        auto end() const noexcept -> std::array<T, Dimensions>::const_iterator;
        auto cbegin() const noexcept -> std::array<T, Dimensions>::const_iterator;
        auto cend() const noexcept -> std::array<T, Dimensions>::const_iterator;

        [[nodiscard]] T magnitude() const noexcept;
        [[nodiscard]] constexpr T magnitude_sqr() const noexcept;

        // Does not check if magnitude is 0
        vector& normalize() noexcept;
        // Check if magnitude is 0
        vector& normalize_safe() noexcept;

        constexpr bool operator==(const vector& other) const noexcept;
        constexpr bool operator!=(const vector& other) const noexcept;
    };

    // Constructors

    template <typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>::vector() noexcept {
        vector_storage<T, Dimensions>::arr.fill(T{0});
    }

    template <typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>::vector(T value) {
        vector_storage<T, Dimensions>::arr.fill(value);
    }

    template <typename T, size_t Dimensions>
    template <typename U> requires (std::is_constructible_v<T, U> || std::is_convertible_v<U, T>)
    constexpr vector<T, Dimensions>::vector(U&& value) {
        vector_storage<T, Dimensions>::arr.fill(static_cast<T>(value));
    }

    template <typename T, size_t Dimensions>
    template <size_t FromDimensions> requires (FromDimensions > Dimensions)
    constexpr vector<T, Dimensions>::vector(const vector<T, FromDimensions>& other) {
        if (std::is_constant_evaluated() || !std::is_trivially_copyable_v<T>) {
            for (size_t i = 0; i < Dimensions; ++i) {
                this->operator[](i) = other[i];
            }
        } else {
            std::memcpy(this, &other, sizeof(T) * Dimensions);
        }
    }

    template <typename T, size_t Dimensions>
    template <size_t FromDimensions> requires (FromDimensions < Dimensions)
    constexpr vector<T, Dimensions>::vector(const vector<T, FromDimensions>& other) {
        vector_storage<T, Dimensions>::arr.fill(T{0});
        if (std::is_constant_evaluated() || !std::is_trivially_copyable_v<T>) {
            for (size_t i = 0; i < FromDimensions; ++i) {
                this->operator[](i) = other[i];
            }
        } else {
            std::memcpy(this, &other, sizeof(T) * FromDimensions);
        }
    }

    template <typename T, size_t Dimensions>
    template <size_t FromDimensions, typename... Args> requires (FromDimensions < Dimensions && Dimensions -
        FromDimensions == sizeof...(Args))
    constexpr vector<T, Dimensions>::vector(const vector<T, FromDimensions>& other, Args&&... args) {
        if (std::is_constant_evaluated() || !std::is_trivially_copyable_v<T>) {
            for (size_t i = 0; i < FromDimensions; ++i) {
                this->operator[](i) = other[i];
            }
        } else {
            std::memcpy(this, &other, sizeof(T) * FromDimensions);
        }
        size_t i = FromDimensions;
        ((this->operator[](i++) = args), ...);
    }

    template <typename T, size_t Dimensions>
    template <typename... Args> requires (sizeof...(Args) == Dimensions)
    constexpr vector<T, Dimensions>::vector(Args&&... args) noexcept
        : vector_storage<T, Dimensions>({static_cast<T>(std::forward<Args>(args))...}) {}

    // Operators

    template <typename T, size_t Dimensions>
    template <typename U>
    constexpr vector<T, Dimensions>::operator vector<U, Dimensions>() const noexcept {
        vector<U, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = static_cast<U>(this->operator[](i));
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr T& vector<T, Dimensions>::operator[](size_t i) noexcept {
        return vector_storage<T, Dimensions>::arr[i];
    }

    template<typename T, size_t Dimensions>
    constexpr const T& vector<T, Dimensions>::operator[](size_t i) const noexcept {
        return vector_storage<T, Dimensions>::arr[i];
    }

    // Functions

    template <typename T, size_t Dimensions>
    constexpr T* vector<T, Dimensions>::data() noexcept {
        return vector_storage<T, Dimensions>::arr.data();
    }

    template <typename T, size_t Dimensions>
    constexpr const T* vector<T, Dimensions>::data() const noexcept {
        return vector_storage<T, Dimensions>::arr.data();
    }

    template <typename T, size_t Dimensions>
    constexpr size_t vector<T, Dimensions>::size() const noexcept {
        return Dimensions;
    }

    template <typename T, size_t Dimensions>
    auto vector<T, Dimensions>::begin() noexcept -> typename std::array<T, Dimensions>::iterator {
        return vector_storage<T, Dimensions>::arr.begin();
    }

    template <typename T, size_t Dimensions>
    auto vector<T, Dimensions>::end() noexcept -> typename std::array<T, Dimensions>::iterator {
        return vector_storage<T, Dimensions>::arr.end();
    }

    template <typename T, size_t Dimensions>
    auto vector<T, Dimensions>::begin() const noexcept -> typename std::array<T, Dimensions>::const_iterator {
        return vector_storage<T, Dimensions>::arr.begin();
    }

    template <typename T, size_t Dimensions>
    auto vector<T, Dimensions>::end() const noexcept -> typename std::array<T, Dimensions>::const_iterator {
        return vector_storage<T, Dimensions>::arr.end();
    }

    template <typename T, size_t Dimensions>
    auto vector<T, Dimensions>::cbegin() const noexcept -> typename std::array<T, Dimensions>::const_iterator {
        return vector_storage<T, Dimensions>::arr.cbegin();
    }

    template <typename T, size_t Dimensions>
    auto vector<T, Dimensions>::cend() const noexcept -> typename std::array<T, Dimensions>::const_iterator {
        return vector_storage<T, Dimensions>::arr.cend();
    }

    template<typename T, size_t Dimensions>
    T vector<T, Dimensions>::magnitude() const noexcept {
        return std::sqrt(magnitude_sqr());
    }

    template<typename T, size_t Dimensions>
    constexpr T vector<T, Dimensions>::magnitude_sqr() const noexcept {
        T result{ 0 };
        for (size_t i = 0; i < Dimensions; ++i) {
            result += this->operator[](i) * this->operator[](i);
        }
        return result;
    }

    template <typename T, size_t Dimensions>
    vector<T, Dimensions>& vector<T, Dimensions>::normalize() noexcept {
        auto mag = magnitude();
        assert(mag != T{0} && "division by zero in unsafe vector::normalize()");
        for (size_t i = 0; i < Dimensions; ++i) {
            this->operator[](i) /= mag;
        }
        return *this;
    }

    template <typename T, size_t Dimensions>
    vector<T, Dimensions>& vector<T, Dimensions>::normalize_safe() noexcept {
        auto mag = magnitude();
        if (mag != T{0}) {
            for (size_t i = 0; i < Dimensions; ++i) {
                this->operator[](i) /= mag;
            }
        }
        return *this;
    }

    template <typename T, size_t Dimensions>
    constexpr bool vector<T, Dimensions>::operator==(const vector& other) const noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            if (!is_equal(this->operator[](i), other[i])) {
                return false;
            }
        }
        return true;
    }

    template <typename T, size_t Dimensions>
    constexpr bool vector<T, Dimensions>::operator!=(const vector& other) const noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            if (!is_equal(this->operator[](i), other[i])) {
                return true;
            }
        }
        return false;
    }

    // Arithmetics

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator-(const vector<T, Dimensions>& rhs) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = -rhs[i];
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator+(const vector<T, Dimensions>& lhs, const vector<T, Dimensions>& rhs) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = lhs[i] + rhs[i];
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator+(const vector<T, Dimensions>& lhs, T val) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = lhs[i] + val;
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>& operator+=(vector<T, Dimensions>& lhs, T val) noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            lhs[i] += val;
        }
        return lhs;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>& operator+=(vector<T, Dimensions>& lhs, const vector<T, Dimensions>& rhs) noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            lhs[i] += rhs[i];
        }
        return lhs;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator-(const vector<T, Dimensions>& lhs, T val) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = lhs[i] - val;
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator-(const vector<T, Dimensions>& lhs, const vector<T, Dimensions>& rhs) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = lhs[i] - rhs[i];
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>& operator-=(vector<T, Dimensions>& lhs, T val) noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            lhs[i] -= val;
        }
        return lhs;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>& operator-=(vector<T, Dimensions>& lhs, const vector<T, Dimensions>& rhs) noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            lhs[i] -= rhs[i];
        }
        return lhs;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator*(const vector<T, Dimensions>& lhs, T val) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = lhs[i] * val;
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator*(T val, const vector<T, Dimensions>& rhs) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = val * rhs[i];
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator*(const vector<T, Dimensions>& lhs, const vector<T, Dimensions>& rhs) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = lhs[i] * rhs[i];
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>& operator*=(vector<T, Dimensions>& lhs, T val) noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            lhs[i] *= val;
        }
        return lhs;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator/(const vector<T, Dimensions>& lhs, T val) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = lhs[i] / val;
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions> operator/(T val, const vector<T, Dimensions>& rhs) noexcept {
        vector<T, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = val / rhs[i];
        }
        return result;
    }

    template<typename T, size_t Dimensions>
    constexpr vector<T, Dimensions>& operator/=(vector<T, Dimensions>& lhs, T val) noexcept {
        for (size_t i = 0; i < Dimensions; ++i) {
            lhs[i] /= val;
        }
        return lhs;
    }

    template<typename Stream, typename T, size_t Dimensions>
    Stream& operator<<(Stream& os, const vector<T, Dimensions>& v) {
        os << "vec" << Dimensions << "(";
        bool comma = false;
        for (size_t i = 0; i < Dimensions; ++i) {
            if (comma) {
                os << ", ";
            }
            comma = true;
            os << v[i];
        }
        return os << ")";
    }

}

namespace std {

    template <typename T, std::size_t Dimensions>
    struct hash<math::impl::vector<T, Dimensions>> {
        std::size_t operator()(const math::impl::vector<T, Dimensions>& vector) const {
            return std::hash<std::array<T, Dimensions>>{}(vector.arr);
        }
    };

}
