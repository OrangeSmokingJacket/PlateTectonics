#pragma once

namespace utils {

    [[noreturn]] inline void unreachable() {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_unreachable();
#elif defined(_MSC_VER)
        __assume(false);
#else
        static_assert(false && "unhandles compiler in unreachable()");
#endif
    }

}