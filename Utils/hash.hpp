#pragma once

#include <array>
#include <cstdint>
#include <memory>

namespace utils {

    void hash_combine(size_t& seed, size_t h2);

}

namespace std {

    template <typename T, std::size_t N>
    struct hash<std::array<T, N>> {
        std::size_t operator()(const std::array<T, N>& arr) const {
            std::size_t seed = 0;
            for (const auto& element : arr) {
                utils::hash_combine(seed, std::hash<T>{}(element));
            }
            return seed;
        }
    };

}