#pragma once

#include <type_traits>

namespace utils {

    template <typename T>
    requires(std::is_enum_v<T>)
    constexpr std::underlying_type_t<T> to_underlying_type(T enum_val) {
        return static_cast<std::underlying_type_t<T>>(enum_val);
    }

    template <typename T>
    requires(std::is_enum_v<T>)
    constexpr T from_underlying_type(std::underlying_type_t<T> val) {
        return static_cast<T>(val);
    }

} // namespace utils
