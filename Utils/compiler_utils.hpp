#pragma once

#include <cassert>

namespace utils {

    [[noreturn]] inline void unreachable() {
        assert(false && "reached to unreachable case");
#if defined(__GNUC__) || defined(__clang__)
        __builtin_unreachable();
#elif defined(_MSC_VER)
        __assume(false);
#else
        static_assert(false && "unhandled compiler in unreachable()");
#endif
    }

} // namespace utils
