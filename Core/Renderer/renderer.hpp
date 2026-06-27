#pragma once

#include "buffer_objects.hpp"

#include "variant_shader_program.hpp"

#include <Core/Engine/Assets/Mesh/mesh.hpp>
#include <Core/Engine/Assets/Material/material.hpp>

#include <unordered_map>

namespace Core {

    struct CameraMatrices {
        alignas(16) math::mat4x4<float> view_matrix;
        alignas(16) math::mat4x4<float> perspective_matrix;
    };

    class RenderPipeline {
    public:
        explicit RenderPipeline(uint32_t buffers_capacity);

        void Init();
        void Cleanup();
        void Unbind();

        void BindCameraMatrices(const CameraMatrices& matrices);
        void SetModelMatrix(const math::mat4x4<float>& model);
        void CreateUnlitShaderVariant(ShaderGenerator::Unlit::ShaderVariance variance);
        // Regular pass
        void BindMesh(const Engine::Assets::Mesh* mesh, const Engine::Assets::MaterialUnlit* material);
        // TODO: submeshes
        void DrawElements(const Engine::Assets::Mesh* mesh, const Engine::Assets::MaterialUnlit* material) const;
        // Slice pass
        void BindSlice(Engine::Assets::AssetRef<Engine::Assets::Texture3Df> sample, const math::vec3<float>& bounds, const math::vec2<float>& resolution, float plain_offset = 0);
        void DrawSlice();

        // Debug pass
        void DebugDrawAll();
        void DebugDrawBox(const math::vec3<float>& center, const math::vec3<float>& extent, Engine::Assets::SColor color, bool persistent);
        void DebugDrawLine(const math::vec3<float>& start, const math::vec3<float>& end, Engine::Assets::SColor color, bool persistent);
        void DebugDrawArrow(const math::vec3<float>& start, const math::vec3<float>& end, float head_size, Engine::Assets::SColor color, bool persistent);

    private:
        struct DebugLine {
            math::vec3<float> start;
            math::vec3<float> end;
        };
        void Reset();
        static void SetBlendingMode(Engine::Assets::BlendMode mode);
        static void SetFaceCulling(Engine::Assets::FaceCulling mode);

        void DebugDrawPrimitive(std::vector<DebugLine>&& lines, Engine::Assets::SColor color, bool persistent);
        // Debug draw data
        std::unordered_map<Engine::Assets::SColor, std::vector<DebugLine>> static_debug_draws_;
        std::unordered_map<Engine::Assets::SColor, std::vector<DebugLine>> per_frame_debug_draws_;

        // Slice mode
        ShaderProgram slice_program_;

        // Common Renderer data
        VertexArrayObject VAO_;
        std::array<VertexBufferObject, Engine::Assets::VertexAttributesCount> VBOs_;
        ElementBufferObject EBO_;
        UniformBufferObject camera_matrices_UBO_;
        TextureObject texture_id_;
        ShaderGenerator::Unlit::VariantShaderProgram unlit_programs_;
        uint32_t bounded_attributes_count_ = 0;
        GLuint bounded_program_id_ = GL_INVALID_INDEX;

    };

}
