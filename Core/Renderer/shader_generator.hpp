#pragma once

#include <Core/Engine/Assets/Material/material.hpp>
#include <Core/Filesystem/filesystem.hpp>

namespace Core::ShaderGenerator {

    namespace Unlit {

        struct ShaderVariance {
            explicit ShaderVariance() = default;
            explicit ShaderVariance(const Engine::Assets::Unlit_MaterialParameters& parameters);
            bool use_vertex_color = false;
            bool use_texture = false;
            bool use_alpha_clipping = false;

            size_t GenerateVertexPermutation() const noexcept;
            size_t GenerateFragmentPermutation() const noexcept;
        };

        Filesystem::buffer_t GenerateVertexShaderHeader(ShaderVariance options);
        Filesystem::buffer_t GenerateFragmentShaderHeader(ShaderVariance options);

    } // namespace Unlit

} // namespace Core::ShaderGenerator
