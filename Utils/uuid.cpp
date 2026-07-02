#include "uuid.hpp"

#include <format>

namespace utils {

    std::string UUIDv4::to_string() const { return std::format("{:x}{:x}", upper_, lower_); }

    UUIDv4 GenerateUUIDv4() {
        thread_local std::random_device rd;
        thread_local std::mt19937_64 gen(rd());
        uint64_t upper = (gen() & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
        uint64_t lower = (gen() & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
        return {lower, upper};
    }

} // namespace utils
