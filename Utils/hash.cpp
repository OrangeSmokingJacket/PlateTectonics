#include "hash.hpp"

namespace utils {

    void hash_combine(size_t& seed, size_t hash_val) { seed ^= hash_val + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

} // namespace utils
