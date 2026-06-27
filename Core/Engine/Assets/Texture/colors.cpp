#include "colors.hpp"

namespace Core::Engine::Assets {

    Color FromSColor(const SColor& input_color) {
        return static_cast<Color>(input_color) / 255.0f;
    }

    SColor FromColor(const Color& input_color) {
        return static_cast<SColor>(input_color * 255.0f);
    }

}
