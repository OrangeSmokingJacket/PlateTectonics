#pragma once

#include "shader_program.hpp"
#include "shader_generator.hpp"

#include <unordered_map>

namespace Core::ShaderGenerator {

    namespace Unlit {

        class VariantShaderProgram {
        public:
            explicit VariantShaderProgram(const std::filesystem::path& vs_path, const std::filesystem::path& fs_path);
            //! creating ShaderProgram is expensive
            ShaderID CreateVariant(ShaderVariance variance);
            bool IsVariantCompiled(ShaderVariance variance) const noexcept;
            const ShaderProgram& BindVariant(ShaderVariance variance);
            void Unbind() const noexcept;

        private:
            std::filesystem::path vs_path_;
            std::filesystem::path fs_path_;
            std::unordered_map<size_t, VertexShaderPtr> vs_variants_;
            std::unordered_map<size_t, FragmentShaderPtr> fs_variants_;
            std::unordered_map<size_t, ShaderProgram> program_variants_;
        };

    }

}
