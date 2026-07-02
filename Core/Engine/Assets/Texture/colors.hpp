#pragma once

#include "texel_data.hpp"
#include <cstdint>

namespace Core::Engine::Assets {

    // TODO: same data structure as Texture<>
    using Color = impl::TexelData<TexelType::float32, TexelFormat::RGBA>;
    using SColor = impl::TexelData<TexelType::uint8, TexelFormat::RGBA>;

    Color FromSColor(const SColor& input_color);
    SColor FromColor(const Color& input_color);

    namespace Colors {

        constexpr Color White{1.0f, 1.0f, 1.0f, 1.0f};
        constexpr Color Black{0.0f, 0.0f, 0.0f, 1.0f};

        constexpr Color Red{1.0f, 0.0f, 0.0f, 1.0f};
        constexpr Color Green{0.0f, 1.0f, 0.0f, 1.0f};
        constexpr Color Blue{0.0f, 0.0f, 1.0f, 1.0f};

        constexpr Color Yellow{1.0f, 1.0f, 0.0f, 1.0f};
        constexpr Color Magenta{1.0f, 0.0f, 1.0f, 1.0f};
        constexpr Color Cyan{0.0f, 1.0f, 1.0f, 1.0f};

    } // namespace Colors

    namespace SColors {

        constexpr SColor White{uint8_t{255}, uint8_t{255}, uint8_t{255}, uint8_t{255}};
        constexpr SColor Black{uint8_t{0}, uint8_t{0}, uint8_t{0}, uint8_t{255}};

        constexpr SColor Red{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}};
        constexpr SColor Green{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}};
        constexpr SColor Blue{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}};

        constexpr SColor Yellow{uint8_t{255}, uint8_t{255}, uint8_t{0}, uint8_t{255}};
        constexpr SColor Magenta{uint8_t{255}, uint8_t{0}, uint8_t{255}, uint8_t{255}};
        constexpr SColor Cyan{uint8_t{0}, uint8_t{255}, uint8_t{255}, uint8_t{255}};

    } // namespace SColors

} // namespace Core::Engine::Assets
