#pragma once

#include <cereal/access.hpp>

#include <random>
#include <type_traits>

namespace utils {

    class UUIDv4 {
    public:
        constexpr UUIDv4() noexcept;
        constexpr UUIDv4(uint64_t lower, uint64_t upper) noexcept;
        constexpr UUIDv4(const UUIDv4&) noexcept = default;
        constexpr UUIDv4& operator=(const UUIDv4&) noexcept = default;
        constexpr UUIDv4(UUIDv4&&) noexcept = default;
        constexpr UUIDv4& operator=(UUIDv4&&) noexcept = default;
        constexpr ~UUIDv4() noexcept = default;

        constexpr bool operator==(const UUIDv4&) const noexcept;
        constexpr bool operator!=(const UUIDv4&) const noexcept;

        constexpr uint64_t Lower() const noexcept;
        constexpr uint64_t Upper() const noexcept;

        static constexpr UUIDv4 InvalidUUID();

        friend class cereal::access;

        template <class Archive>
        void save(Archive& ar) const {
            ar(lower_, upper_);
        }

        template <class Archive>
        void load(Archive& ar) {
            ar(lower_, upper_);
        }

        std::string to_string() const;

    private:
        uint64_t lower_;
        uint64_t upper_;
    };

    static_assert(std::is_trivially_copyable_v<UUIDv4>);
    static_assert(std::is_trivially_destructible_v<UUIDv4>);

    constexpr UUIDv4::UUIDv4() noexcept
        : lower_(0)
        , upper_(0) {}

    constexpr UUIDv4::UUIDv4(uint64_t lower, uint64_t upper) noexcept
        : lower_(lower)
        , upper_(upper) {}

    constexpr bool UUIDv4::operator==(const UUIDv4& other) const noexcept {
        return lower_ == other.lower_ && upper_ == other.upper_;
    }

    constexpr bool UUIDv4::operator!=(const UUIDv4& other) const noexcept {
        return lower_ != other.lower_ || upper_ != other.upper_;
    }

    constexpr uint64_t UUIDv4::Lower() const noexcept { return lower_; }
    constexpr uint64_t UUIDv4::Upper() const noexcept { return upper_; }
    constexpr UUIDv4 UUIDv4::InvalidUUID() { return UUIDv4(); }

    UUIDv4 GenerateUUIDv4();


}

namespace std {
    template <>
    struct hash<utils::UUIDv4> {
        size_t operator()(const utils::UUIDv4& uuid) const noexcept {
            size_t h1 = std::hash<uint64_t>{}(uuid.Lower());
            size_t h2 = std::hash<uint64_t>{}(uuid.Upper());

            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
}