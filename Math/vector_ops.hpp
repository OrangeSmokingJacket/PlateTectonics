#pragma once

#include "impl/vector.hpp"

#include <algorithm>

namespace math {

    template <typename T, size_t Dimensions>
    [[nodiscard]] constexpr T dot_product(const impl::vector<T, Dimensions>& lhs,
                                          const impl::vector<T, Dimensions>& rhs) noexcept {
        T result{0};
        for (size_t i = 0; i < Dimensions; ++i) {
            result += lhs[i] * rhs[i];
        }
        return result;
    }

    template <typename T>
    [[nodiscard]] constexpr impl::vector<T, 3> cross_product(const impl::vector<T, 3>& lhs,
                                                             const impl::vector<T, 3>& rhs) noexcept {
        return impl::vector<T, 3>{lhs[1] * rhs[2] - lhs[2] * rhs[1],
                                  lhs[2] * rhs[0] - lhs[0] * rhs[2],
                                  lhs[0] * rhs[1] - lhs[1] * rhs[0]};
    }

    template <typename T, size_t Dimensions>
    [[nodiscard]] constexpr T magnitude_sqr(const impl::vector<T, Dimensions>& v) noexcept {
        return dot_product(v, v);
    }

    template <typename T, size_t Dimensions>
    [[nodiscard]] T magnitude(const impl::vector<T, Dimensions>& v) noexcept {
        return std::sqrt(magnitude_sqr(v));
    }

    template <typename T, size_t Dimensions>
    [[nodiscard]] impl::vector<T, Dimensions> normalize(const impl::vector<T, Dimensions>& v) noexcept {
        impl::vector<T, Dimensions> result{};
        auto mag = magnitude(v);
        assert(mag != T{0} && "division by zero in unsafe normalize(vector)");
        for (size_t i = 0; i < Dimensions; ++i) {
            result[i] = v[i] / mag;
        }
        return result;
    }

    template <typename T, size_t Dimensions>
    [[nodiscard]] impl::vector<T, Dimensions> normalize_safe(const impl::vector<T, Dimensions>& v) noexcept {
        impl::vector<T, Dimensions> result = impl::vector<T, Dimensions>::zero_vector();
        auto mag = magnitude(v);
        if (mag != T{0}) {
            for (size_t i = 0; i < Dimensions; ++i) {
                result[i] = v[i] / mag;
            }
        }
        return result;
    }

    template <typename T, size_t Dimensions>
    requires(Dimensions > 1)
    [[nodiscard]] constexpr impl::vector<T, Dimensions>
    best_orthogonal(const impl::vector<T, Dimensions>& input) noexcept {
        auto index = static_cast<size_t>(
            std::distance(input.begin(),
                          std::min_element(input.begin(), input.end(), [](const auto& lhs, const auto& rhs) {
                              return std::abs(lhs) < std::abs(rhs);
                          })));
        impl::vector<T, Dimensions> result{};
        result[index] = T{1}; // does not change sign to avoid dealing with unsigned T
        return result;
    }

} // namespace math
