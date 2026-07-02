#pragma once

#include <array>
#include <cstdint>
#include <utility>

namespace utils {

    template <typename T, typename... Args, size_t... Indices>
    std::array<T, sizeof...(Indices)> InitializeAll(std::index_sequence<Indices...>, Args&&... args) {
        return {(void(Indices), T(args...))...};
    }

} // namespace utils
