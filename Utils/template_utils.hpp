#pragma once

#include <utility>

namespace utils {

    template <size_t, typename F, size_t... I>
    void unrolled_for_impl(F&& f, std::index_sequence<I...>) {
        (f(std::integral_constant<size_t, I>{}), ...);
    }

    template <size_t N, typename F>
    void unrolled_for(F&& f) {
        unrolled_for_impl<N>(std::forward<F>(f), std::make_index_sequence<N>{});
    }
} // namespace utils
