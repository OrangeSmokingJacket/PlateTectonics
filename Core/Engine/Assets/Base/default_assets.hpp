#pragma once

#include "base.hpp"

#include <array>
#include <string_view>

namespace Core::Engine::Assets {

    struct KeyValue {
        std::string_view Name;
        AssetID ID;
    };

    template <size_t Size>
    class asset_map_t {
    public:
        constexpr asset_map_t(std::array<KeyValue, Size> init)
            : data_(init) {}

        constexpr std::string_view GetName(AssetID ID) const {
            for (const auto& entry : data_) {
                if (entry.ID == ID) {
                    return entry.Name;
                }
            }
            throw std::out_of_range("There is no default asset with ID: " + ID.to_string());
        }

        constexpr AssetID GetID(std::string_view Name) const {
            for (const auto& entry : data_) {
                if (entry.Name == Name) {
                    return entry.ID;
                }
            }
            throw std::out_of_range("There is no default asset with Name: " + std::string(Name));
        }

    private:
        std::array<KeyValue, Size> data_;
    };

    // This is a temp
    constexpr asset_map_t DefaultAssets(std::array{
        KeyValue{"DefaultTexture", AssetID(0x8DAA9967E548EAD5, 0xBC4FEAE26472413F)},
        KeyValue{"CubeUV", AssetID(0xA5534F53D183DBEB, 0x0874747A1EAF4E4D)}});

} // namespace Core::Engine::Assets
