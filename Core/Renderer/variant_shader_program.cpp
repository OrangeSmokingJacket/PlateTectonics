#include "variant_shader_program.hpp"

namespace Core::ShaderGenerator {

    namespace Unlit {

        VariantShaderProgram::VariantShaderProgram(const std::filesystem::path& vs_path,
                                                                           const std::filesystem::path& fs_path)
            : vs_path_(vs_path)
            , fs_path_(fs_path) {}

        ShaderID VariantShaderProgram::CreateVariant(ShaderVariance variance) {
            auto vs_permutation = variance.GenerateVertexPermutation();
            auto fs_permutation = variance.GenerateFragmentPermutation();

            if (auto it = program_variants_.find(fs_permutation); it != program_variants_.end()) {
                return it->second.ProgramId();
            }

            VertexShaderPtr vs;
            FragmentShaderPtr fs;

            if (auto vs_it = vs_variants_.find(vs_permutation); vs_it != vs_variants_.end()) {
                vs = vs_it->second;
            } else {
                auto header = GenerateVertexShaderHeader(variance);
                vs = std::make_shared<VertexShader>(CompileShader<ShaderType::Vertex>(header + Filesystem::ReadFullFile(vs_path_)));
                vs_variants_.emplace(vs_permutation, vs);
            }
            if (auto fs_it = fs_variants_.find(fs_permutation); fs_it != fs_variants_.end()) {
                fs = fs_it->second;
            } else {
                auto header = GenerateFragmentShaderHeader(variance);
                fs = std::make_shared<FragmentShader>(CompileShader<ShaderType::Fragment>(header + Filesystem::ReadFullFile(fs_path_)));
                fs_variants_.emplace(fs_permutation, fs);
            }

            auto program = ShaderProgram{std::move(vs), std::move(fs)};
            auto id = program.ProgramId();
            program_variants_.emplace(fs_permutation, std::move(program));
            return id;
        }

        bool VariantShaderProgram::IsVariantCompiled(ShaderVariance variance) const noexcept {
            auto fs_permutation = variance.GenerateFragmentPermutation();
            return program_variants_.contains(fs_permutation);
        }

        const ShaderProgram& VariantShaderProgram::BindVariant(ShaderVariance variance) {
            assert(IsVariantCompiled(variance));
            auto& program = program_variants_.at(variance.GenerateFragmentPermutation());
            program.Bind();
            return program;
        }

        void VariantShaderProgram::Unbind() const noexcept {
            ShaderProgram::Unbind();
        }
    }

}
