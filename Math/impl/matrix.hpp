#pragma once
#include "vector.hpp"

namespace math::impl {

    // TODO: conversion from and to vector

    template <typename T, size_t Columns, size_t Rows>
    struct matrix : std::array<vector<T, Rows>, Columns> {
        constexpr matrix() noexcept { std::array<vector<T, Rows>, Columns>::fill(vector<T, Rows>{}); }

        constexpr matrix(const matrix&) noexcept = default;
        constexpr matrix(matrix&&) noexcept = default;
        constexpr matrix& operator=(const matrix&) noexcept = default;
        constexpr matrix& operator=(matrix&&) noexcept = default;

        template <typename... Args>
        requires(sizeof...(Args) == Columns)
        explicit constexpr matrix(Args&&... args) noexcept
            : std::array<vector<T, Rows>, Columns>({std::forward<Args>(args)...}) {}

        static constexpr size_t cardinality = Columns * Rows;

        constexpr T* data() noexcept { return std::array<vector<T, Rows>, Columns>::data()->data(); }
        constexpr const T* data() const noexcept { return std::array<vector<T, Rows>, Columns>::data()->data(); }

        static constexpr matrix identity() noexcept requires(Columns == Rows)
        {
            matrix result{};
            for (size_t i = 0; i < Columns; ++i) {
                result[i][i] = T{1};
            }
            return result;
        }
    };

    template <typename T, size_t Columns, size_t Rows>
    constexpr vector<T, Columns> operator*(const matrix<T, Columns, Rows>& m, const vector<T, Rows>& v) noexcept {
        vector<T, Columns> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Columns; ++j) {
                result[i] += m[j][i] * v[j];
            }
        }
        return result;
    }

    template <typename T, size_t M, size_t K, size_t N>
    constexpr matrix<T, M, N> operator*(const matrix<T, K, N>& m1, const matrix<T, M, K>& m2) noexcept {
        matrix<T, M, N> result{};
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    result[i][j] += m1[k][j] * m2[i][k];
                }
            }
        }
        return result;
    }

    template <typename T, size_t Columns, size_t Rows>
    constexpr matrix<T, Columns, Rows> operator*(const matrix<T, Columns, Rows>& lhs, T val) noexcept {
        matrix<T, Columns, Rows> result;
        for (size_t i = 0; i < matrix<T, Columns, Rows>::cardinality; ++i) {
            result.data()[i] = lhs.data()[i] * val;
        }
        return result;
    }

    template <typename T, size_t Columns, size_t Rows>
    constexpr matrix<T, Columns, Rows> operator*(T val, const matrix<T, Columns, Rows>& rhs) noexcept {
        matrix<T, Columns, Rows> result;
        for (size_t i = 0; i < matrix<T, Columns, Rows>::cardinality; ++i) {
            result.data()[i] = val * rhs.data()[i];
        }
        return result;
    }

    template <typename T, size_t Columns, size_t Rows>
    constexpr matrix<T, Columns, Rows> operator/(const matrix<T, Columns, Rows>& lhs, T val) noexcept {
        matrix<T, Columns, Rows> result;
        for (size_t i = 0; i < matrix<T, Columns, Rows>::cardinality; ++i) {
            result.data()[i] = lhs.data()[i] / val;
        }
        return result;
    }

} // namespace math::impl

namespace std {

    template <typename T, size_t Columns, size_t Rows>
    struct hash<math::impl::matrix<T, Columns, Rows>> {
        std::size_t operator()(const math::impl::matrix<T, Columns, Rows>& matrix) const {
            return std::hash<std::array<math::impl::vector<T, Rows>, Columns>>{}(matrix);
        }
    };

} // namespace std
