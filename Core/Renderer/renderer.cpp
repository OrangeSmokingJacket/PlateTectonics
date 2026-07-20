#include "renderer.hpp"

#include <cassert>

namespace Core {

    template <typename T, size_t... Indices>
    std::array<T, sizeof...(Indices)> MakeArrayFromIndices(uint32_t value, std::index_sequence<Indices...>) {
        return {{((void)Indices, T{value})...}};
    }

    template <typename T, size_t Size>
    std::array<T, Size> MakeArray(uint32_t value) {
        return MakeArrayFromIndices<T>(value, std::make_index_sequence<Size>{});
    }

    static constexpr std::array<GLenum, 3> axis_mapping = {GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R};
    static constexpr std::array<GLenum, 3> dimensions_mapping = {GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D};

    template <size_t Dimensions>
    requires(Dimensions <= 3)
    void SetTextureParameter(Engine::Assets::TextureFiltering filtering) {
        static constexpr auto target = dimensions_mapping[Dimensions - 1];
        using namespace Engine::Assets;

        switch (filtering) {
            case TextureFiltering::Nearest:
                glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                break;
            case TextureFiltering::Linear:
                glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                break;
            default:
                assert(false);
        }
    }

    template <size_t Dimensions>
    requires(Dimensions <= 3)
    void SetTextureParameter(const math::impl::vector<Engine::Assets::TextureWrapping, Dimensions>& wrapping) {
        static constexpr auto target = dimensions_mapping[Dimensions - 1];
        using namespace Engine::Assets;

        for (size_t i = 0; i < Dimensions; ++i) {
            switch (wrapping[i]) {
                case TextureWrapping::Repeat:
                    glTexParameteri(target, axis_mapping[i], GL_REPEAT);
                    break;
                case TextureWrapping::MirroredRepeat:
                    glTexParameteri(target, axis_mapping[i], GL_MIRRORED_REPEAT);
                    break;
                case TextureWrapping::ClampToBorder:
                    glTexParameteri(target, axis_mapping[i], GL_CLAMP_TO_BORDER);
                    break;
                case TextureWrapping::ClampToEdge:
                    glTexParameteri(target, axis_mapping[i], GL_CLAMP_TO_EDGE);
                    break;
                default:
                    assert(false);
            }
        }
    }

    RenderPipeline::RenderPipeline(uint32_t buffers_capacity)
        : slice_program_(std::make_shared<VertexShader>(shaders_dir / "slice_vs.glsl"),
                         std::make_shared<FragmentShader>(shaders_dir / "slice_fs.glsl"))
        , VAO_()
        , VBOs_(MakeArray<VertexBufferObject, Engine::Assets::VertexAttributesCount>(buffers_capacity))
        , EBO_(buffers_capacity)
        , camera_matrices_UBO_(sizeof(CameraMatrices))
        , unlit_programs_(shaders_dir / "global_unlit_vs_body.glsl", shaders_dir / "global_unlit_fs_body.glsl") {
        slice_program_.Bind();
        GLuint UBI = glGetUniformBlockIndex(slice_program_.ProgramId(), "CameraMatrices");
        assert(UBI != GL_INVALID_INDEX);
        slice_program_.Unbind();
    }

    void RenderPipeline::Init() { VAO_.Bind(); }

    void RenderPipeline::Cleanup() {
        VAO_.Unbind();
        texture_id_.Unbind();
    }

    void RenderPipeline::Unbind() {
        glUseProgram(0);
        bounded_program_id_ = GL_INVALID_INDEX;
    }

    void RenderPipeline::BindCameraMatrices(const CameraMatrices& matrices) {
        camera_matrices_UBO_.BindBase(0);

        camera_matrices_UBO_.PopulateData(&matrices, sizeof(CameraMatrices));
        camera_matrices_UBO_.Unbind();
    }

    void RenderPipeline::SetModelMatrix(const math::mat4x4<float>& model) {
        GLint uniform_var_id = glGetUniformLocation(bounded_program_id_, "model_matrix");
        assert(uniform_var_id >= 0 && "glGetUniformLocation did not find a value by name");
        glUniformMatrix4fv(uniform_var_id, 1, GL_FALSE, model.data());
    }

    void RenderPipeline::CreateUnlitShaderVariant(ShaderGenerator::Unlit::ShaderVariance variance) {
        unlit_programs_.CreateVariant(variance);
        const auto& program = unlit_programs_.BindVariant(variance);
        GLuint UBI = glGetUniformBlockIndex(program.ProgramId(), "CameraMatrices");
        assert(UBI != GL_INVALID_INDEX);
        // TODO: keep track of used uniformBlockBindings
        glUniformBlockBinding(program.ProgramId(), UBI, 0);
    }

    void RenderPipeline::BindMesh(const Engine::Assets::Mesh* mesh, const Engine::Assets::MaterialUnlit* material) {
        using namespace Engine::Assets;
        const MeshData& mesh_data = mesh->mesh_data;
        const auto& material_parameters = material->GetParameters();
        assert(mesh_data.positions);

        ShaderGenerator::Unlit::ShaderVariance variance(material_parameters);
        // This will cause stater on the first call
        // TODO: Setup all required variants in initialization
        CreateUnlitShaderVariant(variance);
        const auto& program = unlit_programs_.BindVariant(variance);
        bounded_program_id_ = program.ProgramId();

        Reset();

        EBO_.Bind();
        EBO_.PopulateData(mesh->elements.elements.data.get(), mesh->elements.elements.ByteSize());

        uint32_t attribute_index = 0;

        VBOs_[VertexAttributePosition(VertexAttribute::Position)].Bind();
        VBOs_[VertexAttributePosition(VertexAttribute::Position)].PopulateData(mesh_data.positions.data.get(),
                                                                               mesh_data.positions.ByteSize());
        glEnableVertexAttribArray(attribute_index);
        glVertexAttribPointer(attribute_index, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        ++attribute_index;

        if (variance.use_vertex_color) {
            assert(mesh_data.colors);

            VBOs_[VertexAttributePosition(VertexAttribute::Color)].Bind();
            VBOs_[VertexAttributePosition(VertexAttribute::Color)].PopulateData(mesh_data.colors.data.get(),
                                                                                mesh_data.colors.ByteSize());
            glEnableVertexAttribArray(attribute_index);
            glVertexAttribPointer(attribute_index, 4, GL_UNSIGNED_BYTE, GL_TRUE, 4 * sizeof(uint8_t), (void*)0);
            ++attribute_index;
        }

        if (variance.use_texture) {
            const auto& binding = material_parameters.base_texture;
            assert(binding.uv_index < mesh_data.uvs.size());
            assert(mesh_data.uvs[binding.uv_index]);

            // Bind corresponding UV
            // TODO: Multiple textures will require storing bounded indices to avoid multiple data copy calls
            {
                auto VBO_index = VertexAttributePosition(VertexAttribute::UV0) + binding.uv_index;
                VBOs_[VBO_index].Bind();
                VBOs_[VBO_index].PopulateData(mesh_data.uvs[binding.uv_index].data.get(),
                                              mesh_data.uvs[binding.uv_index].ByteSize());
                glEnableVertexAttribArray(attribute_index);
                glVertexAttribPointer(attribute_index, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
                ++attribute_index;
            }

            // Assign to texture slot 0
            texture_id_.SetActiveUnit(0);
            texture_id_.Bind();

            SetTextureParameter<2>(binding.filtering);
            SetTextureParameter(binding.wrapping);

            // TODO: different texture types

            const auto* texture = binding.texture_ref.GetAsset();
            assert(texture);
            glTexImage2D(GL_TEXTURE_2D,
                         0,
                         GL_RGBA,
                         texture->Width(),
                         texture->Height(),
                         0,
                         GL_RGBA,
                         GL_UNSIGNED_BYTE,
                         texture->Data());
            glGenerateMipmap(GL_TEXTURE_2D);

            {
                GLint uniform_tex = glGetUniformLocation(bounded_program_id_, "diffuse_texture");
                assert(uniform_tex >= 0 && "glGetUniformLocation did not find a value by name");
                glUniform1i(uniform_tex, 0);
            }
            {
                GLint uniform_tiling = glGetUniformLocation(bounded_program_id_, "uv_tiling");
                assert(uniform_tiling >= 0 && "glGetUniformLocation did not find a value by name");
                glUniform2f(uniform_tiling, binding.uv_tiling.x, binding.uv_tiling.y);
            }
            {
                GLint uniform_offset = glGetUniformLocation(bounded_program_id_, "uv_offset");
                assert(uniform_offset >= 0 && "glGetUniformLocation did not find a value by name");
                glUniform2f(uniform_offset, binding.uv_offset.x, binding.uv_offset.y);
            }
        }

        if (variance.use_alpha_clipping) {
            GLint uniform_threshold = glGetUniformLocation(bounded_program_id_, "alpha_threshold");
            assert(uniform_threshold >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform1f(uniform_threshold, material_parameters.alpha_clip_threshold);
        }

        {
            const Color& base_color = material_parameters.base_color;
            GLint uniform_color = glGetUniformLocation(bounded_program_id_, "base_color");
            assert(uniform_color >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform4f(uniform_color, base_color.r, base_color.g, base_color.b, base_color.a);
        }

        bounded_attributes_count_ = attribute_index;
    }

    void RenderPipeline::DrawElements(const Engine::Assets::Mesh* mesh,
                                      const Engine::Assets::MaterialUnlit* material) const {
        const auto& sub_mesh = mesh->elements.sub_meshes[0];

        // TODO: those have to come from the material
        SetBlendingMode(material->GetParameters().blend_mode);
        SetFaceCulling(material->GetParameters().face_culling);

        switch (sub_mesh.element_type) {
            case Engine::Assets::ElementType::Point:
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
                glDrawElements(GL_POINTS,
                               sub_mesh.count,
                               GL_UNSIGNED_INT,
                               reinterpret_cast<const GLvoid*>(sub_mesh.start_index * sizeof(uint32_t)));
                break;
            case Engine::Assets::ElementType::Edge:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glDrawElements(GL_LINES,
                               sub_mesh.count,
                               GL_UNSIGNED_INT,
                               reinterpret_cast<const GLvoid*>(sub_mesh.start_index * sizeof(uint32_t)));
                break;
            case Engine::Assets::ElementType::Triangle:
                // Override to triangles
                if (material->GetParameters().wireframe_mode) {
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                } else {
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                }
                glDrawElements(GL_TRIANGLES,
                               sub_mesh.count,
                               GL_UNSIGNED_INT,
                               reinterpret_cast<const GLvoid*>(sub_mesh.start_index * sizeof(uint32_t)));
                break;
            default:
                assert(false);
        }
    }

    void RenderPipeline::BindSlice(
        const std::array<Engine::Assets::AssetRef<Engine::Assets::Texture3Df>, 6>& cube_face_samplers,
        const Engine::Assets::Color& core_color,
        const math::vec3<uint32_t>& face_dimensions,
        const math::vec3<uint32_t>& halo,
        float cell_size,
        const math::vec2<float>& resolution,
        Engine::Assets::TextureFiltering filtering,
        float plain_offset) {
        slice_program_.Bind();
        bounded_program_id_ = slice_program_.ProgramId();

        Reset();

        for (size_t i = 0; i < cube_face_samplers.size(); ++i) {
            const auto* texture = cube_face_samplers[i].GetAsset();
            assert(texture);

            slice_textures_[i].Bind();
            slice_textures_[i].SetActiveUnit(i);

            SetTextureParameter<3>(filtering);
            // If everything is right, it will never be used
            // Clamp to edge is the fastest method available
            // SetTextureParameter(
            //     math::vec3<Engine::Assets::TextureWrapping>{Engine::Assets::TextureWrapping::ClampToEdge});
            SetTextureParameter(math::vec3<Engine::Assets::TextureWrapping>{Engine::Assets::TextureWrapping::Repeat});

            glTexImage3D(GL_TEXTURE_3D,
                         0,
                         GL_RGBA,
                         texture->Width(),
                         texture->Height(),
                         texture->Depth(),
                         0,
                         GL_RGBA,
                         GL_FLOAT,
                         texture->Data());
            glGenerateMipmap(GL_TEXTURE_3D);

            slice_textures_[i].Unbind();
        }

        {
            GLint uniform_face_dimensions = glGetUniformLocation(bounded_program_id_, "face_dimensions");
            assert(uniform_face_dimensions >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform3ui(uniform_face_dimensions, face_dimensions.x, face_dimensions.y, face_dimensions.z);
        }
        {
            GLint uniform_halo = glGetUniformLocation(bounded_program_id_, "halo");
            assert(uniform_halo >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform3ui(uniform_halo, halo.x, halo.y, halo.z);
        }
        {
            GLint uniform_cell_size = glGetUniformLocation(bounded_program_id_, "cell_size");
            assert(uniform_cell_size >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform1f(uniform_cell_size, cell_size);
        }
        {
            GLint uniform_resolution = glGetUniformLocation(bounded_program_id_, "resolution");
            assert(uniform_resolution >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform2f(uniform_resolution, resolution.x, resolution.y);
        }
        {
            GLint uniform_plain_offset = glGetUniformLocation(bounded_program_id_, "plain_offset");
            assert(uniform_plain_offset >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform1f(uniform_plain_offset, plain_offset);
        }
        {
            GLint uniform_color = glGetUniformLocation(bounded_program_id_, "core_color");
            assert(uniform_color >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform4f(uniform_color, core_color.r, core_color.g, core_color.b, core_color.a);
        }
        {
            GLint uniform_tex = glGetUniformLocation(bounded_program_id_, "sample_textures");
            GLint samplers_indices[6] = {0, 1, 2, 3, 4, 5};
            assert(uniform_tex >= 0 && "glGetUniformLocation did not find a value by name");
            glUniform1iv(uniform_tex, 6, samplers_indices);
        }
    }

    void RenderPipeline::DrawSlice() {
        SetBlendingMode(Engine::Assets::BlendMode::Opaque);
        SetFaceCulling(Engine::Assets::FaceCulling::None);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    void RenderPipeline::ReserveLineSpace(Engine::Assets::SColor color, size_t count, bool persistent) {
        auto reserve_space = [](std::unordered_map<Engine::Assets::SColor, std::vector<DebugLine>>& storage,
                                Engine::Assets::SColor color,
                                size_t count) {
            if (auto it = storage.find(color); it == storage.end()) {
                storage.emplace(color, std::vector<DebugLine>{}).first->second.reserve(count);
            } else {
                it->second.reserve(count);
            }
        };

        if (persistent) {
            reserve_space(static_debug_draws_, color, count);
        } else {
            reserve_space(per_frame_debug_draws_, color, count);
        }
    }

    void RenderPipeline::DebugDrawAll() {
        using namespace Core::Engine::Assets;
        // Debug draw always uses default unlit variance
        static const ShaderGenerator::Unlit::ShaderVariance variance{};

        // This will cause stater on the first call
        // TODO: Setup all required variants in initialization
        CreateUnlitShaderVariant(variance);
        const auto& program = unlit_programs_.BindVariant(variance);
        bounded_program_id_ = program.ProgramId();

        VAO_.Bind();
        Reset();
        VBOs_[VertexAttributePosition(VertexAttribute::Position)].Bind();
        bounded_attributes_count_ = 1;

        SetBlendingMode(BlendMode::Transparent);
        SetFaceCulling(FaceCulling::None);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        SetModelMatrix(math::mat4x4<float>::identity());

        auto draw = [this](const std::unordered_map<SColor, std::vector<DebugLine>>& draw_map) {
            for (const auto& draw_batch : draw_map) {
                VBOs_[VertexAttributePosition(VertexAttribute::Position)].PopulateData(draw_batch.second.data(),
                                                                                       draw_batch.second.size() *
                                                                                           sizeof(DebugLine));
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

                Color base_color = FromSColor(draw_batch.first);
                GLint uniform_color = glGetUniformLocation(bounded_program_id_, "base_color");
                assert(uniform_color >= 0 && "glGetUniformLocation did not find a value by name");
                glUniform4f(uniform_color, base_color.r, base_color.g, base_color.b, base_color.a);

                glDrawArrays(GL_LINES, 0, draw_batch.second.size() * 2);
            }
        };

        draw(static_debug_draws_);
        draw(per_frame_debug_draws_);

        per_frame_debug_draws_.clear();
        VAO_.Unbind();
    }

    void RenderPipeline::DebugDrawBox(const math::vec3<float>& center,
                                      const math::vec3<float>& extent,
                                      Engine::Assets::SColor color,
                                      bool persistent) {
        thread_local std::array<DebugLine, 12> lines;
        lines.fill({center, center});
        // along X
        lines[0].start += math::vec3<float>{extent.x, extent.y, extent.z};
        lines[0].end += math::vec3<float>{-extent.x, extent.y, extent.z};
        lines[1].start += math::vec3<float>{extent.x, extent.y, -extent.z};
        lines[1].end += math::vec3<float>{-extent.x, extent.y, -extent.z};
        lines[2].start += math::vec3<float>{extent.x, -extent.y, extent.z};
        lines[2].end += math::vec3<float>{-extent.x, -extent.y, extent.z};
        lines[3].start += math::vec3<float>{extent.x, -extent.y, -extent.z};
        lines[3].end += math::vec3<float>{-extent.x, -extent.y, -extent.z};
        // along Y
        lines[4].start += math::vec3<float>{extent.x, extent.y, extent.z};
        lines[4].end += math::vec3<float>{extent.x, -extent.y, extent.z};
        lines[5].start += math::vec3<float>{extent.x, extent.y, -extent.z};
        lines[5].end += math::vec3<float>{extent.x, -extent.y, -extent.z};
        lines[6].start += math::vec3<float>{-extent.x, extent.y, extent.z};
        lines[6].end += math::vec3<float>{-extent.x, -extent.y, extent.z};
        lines[7].start += math::vec3<float>{-extent.x, extent.y, -extent.z};
        lines[7].end += math::vec3<float>{-extent.x, -extent.y, -extent.z};
        // along Z
        lines[8].start += math::vec3<float>{extent.x, extent.y, extent.z};
        lines[8].end += math::vec3<float>{extent.x, extent.y, -extent.z};
        lines[9].start += math::vec3<float>{extent.x, -extent.y, extent.z};
        lines[9].end += math::vec3<float>{extent.x, -extent.y, -extent.z};
        lines[10].start += math::vec3<float>{-extent.x, extent.y, extent.z};
        lines[10].end += math::vec3<float>{-extent.x, extent.y, -extent.z};
        lines[11].start += math::vec3<float>{-extent.x, -extent.y, extent.z};
        lines[11].end += math::vec3<float>{-extent.x, -extent.y, -extent.z};

        DebugDrawPrimitive(lines, color, persistent);
    }

    void RenderPipeline::DebugDrawLine(const math::vec3<float>& start,
                                       const math::vec3<float>& end,
                                       Engine::Assets::SColor color,
                                       bool persistent) {
        thread_local std::array<DebugLine, 1> line;
        if (start == end) {
            return;
        }
        line[0] = {start, end};
        DebugDrawPrimitive(line, color, persistent);
    }

    void RenderPipeline::DebugDrawArrow(const math::vec3<float>& start,
                                        const math::vec3<float>& end,
                                        float head_size,
                                        Engine::Assets::SColor color,
                                        bool persistent) {
        thread_local DebugLine lines[3];

        if (start == end) {
            return;
        }
        math::vec3<float> direction = end - start;
        direction.normalize();
        // best_orthogonal causes flickering when plain is changing
        // weighted average of 2 smallest axis will help but it does have edgecases where arrow is almost axis alligned
        auto orth = math::best_orthogonal(direction);

        lines[0] = {start, end};
        lines[1] = {end, end + (direction * -head_size) + (orth * head_size)};
        lines[2] = {end, end + (direction * -head_size) + (orth * -head_size)};

        DebugDrawPrimitive(lines, color, persistent);
    }

    void RenderPipeline::Reset() {
        for (uint32_t i = 0; i < bounded_attributes_count_; ++i) {
            glDisableVertexAttribArray(i);
        }
        bounded_attributes_count_ = 0;

        VertexBufferObject::Unbind();
        ElementBufferObject::Unbind();
        TextureObject2D::Unbind();
    }

    void RenderPipeline::SetBlendingMode(Engine::Assets::BlendMode mode) {
        glEnable(GL_DEPTH_TEST);
        if (mode == Engine::Assets::BlendMode::AlphaClipping) {
            glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            glDisable(GL_BLEND);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_TRUE);
        } else if (mode == Engine::Assets::BlendMode::Opaque) {
            glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            glDisable(GL_BLEND);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_TRUE);
        } else {
            glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            glEnable(GL_BLEND);
            glDepthMask(GL_FALSE);
            switch (mode) {
                case Engine::Assets::BlendMode::Transparent:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    break;
                case Engine::Assets::BlendMode::PremultipliedAlpha:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
                    break;
                case Engine::Assets::BlendMode::Add:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_ONE, GL_ONE);
                    break;
                case Engine::Assets::BlendMode::AddWeighted:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                    break;
                case Engine::Assets::BlendMode::Multiply:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_DST_COLOR, GL_ZERO);
                    break;
                case Engine::Assets::BlendMode::Multiply2:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_DST_COLOR, GL_SRC_COLOR);
                    break;
                case Engine::Assets::BlendMode::Screen:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
                    break;
                case Engine::Assets::BlendMode::Subtract:
                    glBlendEquation(GL_FUNC_SUBTRACT);
                    glBlendFunc(GL_ONE, GL_ONE);
                    break;
                case Engine::Assets::BlendMode::Darken:
                    glBlendEquation(GL_MIN);
                    break;
                case Engine::Assets::BlendMode::Lighten:
                    glBlendEquation(GL_MAX);
                    break;
                default:
                    assert(false && "unhandled BlendMode in RenderPipeline::SetBlendingMode");
                    break;
            }
        }
    }

    void RenderPipeline::SetFaceCulling(Engine::Assets::FaceCulling mode) {
        switch (mode) {
            case Engine::Assets::FaceCulling::None:
                glDisable(GL_CULL_FACE);
                break;
            case Engine::Assets::FaceCulling::Back:
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
                break;
            case Engine::Assets::FaceCulling::Front:
                glEnable(GL_CULL_FACE);
                glCullFace(GL_FRONT);
                break;
            case Engine::Assets::FaceCulling::Both:
                glEnable(GL_CULL_FACE);
                glCullFace(GL_FRONT_AND_BACK);
                break;
            default:
                assert(false && "unhandled FaceCulling in RenderPipeline::SetFaceCulling");
                break;
        }
    }

    void RenderPipeline::DebugDrawPrimitive(std::span<DebugLine> lines, Engine::Assets::SColor color, bool persistent) {
        auto insert_draw = [](std::unordered_map<Engine::Assets::SColor, std::vector<DebugLine>>& storage,
                              std::span<DebugLine> lines,
                              Engine::Assets::SColor color) {
            if (auto it = storage.find(color); it == storage.end()) {
                storage.emplace(color, std::vector<DebugLine>{lines.begin(), lines.end()});
            } else {
                it->second.insert(it->second.end(), lines.begin(), lines.end());
            }
        };

        if (persistent) {
            insert_draw(static_debug_draws_, lines, color);
        } else {
            insert_draw(per_frame_debug_draws_, lines, color);
        }
    }

} // namespace Core
