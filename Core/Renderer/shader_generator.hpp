#pragma once

#include <Core/Filesystem/filesystem.hpp>
#include <Core/Engine/Assets/Material/material.hpp>

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

    }

}
