#include "shader_generator.hpp"

namespace Core::ShaderGenerator {

    namespace Unlit {

        ShaderVariance::ShaderVariance(const Engine::Assets::Unlit_MaterialParameters& parameters)
            : use_vertex_color(parameters.use_vertex_color)
            , use_texture(parameters.base_texture.texture_ref.HasAsset() && !parameters.wireframe_mode)
            , use_alpha_clipping(parameters.blend_mode == Engine::Assets::BlendMode::AlphaClipping) {}

        size_t ShaderVariance::GenerateVertexPermutation() const noexcept {
            size_t result = 0;
            if (use_vertex_color) {
                result |= size_t{1} << 0;
            }
            if (use_texture) {
                result |= size_t{1} << 1;
            }
            return result;
        }

        size_t ShaderVariance::GenerateFragmentPermutation() const noexcept {
            size_t result = GenerateVertexPermutation();
            if (use_alpha_clipping) {
                result |= size_t{1} << 2;
            }
            return result;
        }

        Filesystem::buffer_t GenerateVertexShaderHeader(ShaderVariance options) {
            Filesystem::buffer_t result;
            result.append("#version 460 core\r\n\r\n");
            if (options.use_vertex_color) {
                result.append("#define USE_VERTEX_COLOR\r\n");
            }
            if (options.use_texture) {
                result.append("#define USE_TEXTURE\r\n");
            }
            result.append("\r\n");
            return result;
        }

        Filesystem::buffer_t GenerateFragmentShaderHeader(ShaderVariance options) {
            Filesystem::buffer_t result;
            result.append("#version 460 core\r\n\r\n");
            if (options.use_vertex_color) {
                result.append("#define USE_VERTEX_COLOR\r\n");
            }
            if (options.use_texture) {
                result.append("#define USE_TEXTURE\r\n");
            }
            if (options.use_alpha_clipping) {
                result.append("#define USE_ALPHA_CLIPPING\r\n");
            }
            result.append("\r\n");
            return result;
        }

    } // namespace Unlit

} // namespace Core::ShaderGenerator
