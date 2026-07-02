#pragma once

#include "impl/matrix.hpp"

namespace math {

    template <typename T, size_t Columns, size_t Rows>
    [[nodiscard]] constexpr impl::matrix<T, Rows, Columns>
    transpose(const impl::matrix<T, Columns, Rows>& mat) noexcept {
        impl::matrix<T, Rows, Columns> result;
        for (size_t i = 0; i < Columns; ++i) {
            for (size_t j = 0; j < Rows; ++j) {
                result[j][i] = mat[i][j];
            }
        }
        return result;
    }

    template <typename T, size_t Columns, size_t Rows>
    [[nodiscard]] constexpr impl::matrix<T, Columns - 1, Rows - 1>
    sub_matrix(const impl::matrix<T, Columns, Rows>& mat, size_t column, size_t row) {
        impl::matrix<T, Columns - 1, Rows - 1> result;
        for (size_t i = 0; i < Columns - 1; ++i) {
            size_t original_i = i < column ? i : i + 1;
            for (size_t j = 0; j < Rows - 1; ++j) {
                size_t original_j = j < row ? j : j + 1;
                result[i][j] = mat[original_i][original_j];
            }
        }
        return result;
    }

    template <typename T>
    [[nodiscard]] constexpr T determinant(const impl::matrix<T, 2, 2>& mat) {
        return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
    }

    template <typename T, size_t Dimensions>
    requires(Dimensions > 2)
    [[nodiscard]] constexpr T determinant(const impl::matrix<T, Dimensions, Dimensions>& mat) {
        T result{0};
        for (size_t i = 0; i < Dimensions; ++i) {
            auto term = mat[i][0] * determinant(sub_matrix(mat, i, 0));
            result += (i % 2 == 0) ? term : -term;
        }
        return result;
    }

    template <typename T, size_t Dimensions>
    [[nodiscard]] constexpr impl::matrix<T, Dimensions, Dimensions>
    cofactor(const impl::matrix<T, Dimensions, Dimensions>& mat) {
        impl::matrix<T, Dimensions, Dimensions> result;
        for (size_t i = 0; i < Dimensions; ++i) {
            for (size_t j = 0; j < Dimensions; ++j) {
                auto term = determinant(sub_matrix(mat, i, j));
                result[i][j] = (i % 2 == 0) ? term : -term;
            }
        }
        return result;
    }

    template <typename T>
    [[nodiscard]] constexpr impl::matrix<T, 2, 2> adjugate(const impl::matrix<T, 2, 2>& mat) {
        return impl::matrix<T, 2, 2>{impl::vector<T, 2>{mat[0][0], -mat[0][1]},
                                     impl::vector<T, 2>{-mat[1][0], mat[1][1]}};
    }

    template <typename T, size_t Dimensions>
    [[nodiscard]] constexpr impl::matrix<T, Dimensions, Dimensions>
    adjugate(const impl::matrix<T, Dimensions, Dimensions>& mat) {
        return transpose(cofactor(mat));
    }

    template <typename T, size_t Dimensions>
    [[nodiscard]] constexpr impl::matrix<T, Dimensions, Dimensions>
    inverse(const impl::matrix<T, Dimensions, Dimensions>& mat) {
        return adjugate(mat) / determinant(mat);
    }

} // namespace math
