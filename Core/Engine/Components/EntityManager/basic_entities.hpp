#pragma once

#include "entity_manager.hpp"

#include <Core/Engine/Assets/Base/default_assets.hpp>
#include <Core/Engine/Components/MeshRenderer/mesh_renderer.hpp>
#include <Core/Engine/Components/Transform/transform.hpp>

namespace Core::Engine::Assets {

    namespace WireframeCube {

        static constexpr std::initializer_list<VertexPosition> positions = {
            VertexPosition{-0.5f, -0.5f, -0.5f},
            VertexPosition{-0.5f, -0.5f, 0.5f},
            VertexPosition{-0.5f, 0.5f, -0.5f},
            VertexPosition{-0.5f, 0.5f, 0.5f},
            VertexPosition{0.5f, -0.5f, -0.5f},
            VertexPosition{0.5f, -0.5f, 0.5f},
            VertexPosition{0.5f, 0.5f, -0.5f},
            VertexPosition{0.5f, 0.5f, 0.5f},
        };

        static constexpr std::array edges{
            EdgeIndices{0, 1},
            EdgeIndices{0, 2},
            EdgeIndices{2, 3},
            EdgeIndices{1, 3},
            EdgeIndices{4, 5},
            EdgeIndices{4, 6},
            EdgeIndices{6, 7},
            EdgeIndices{5, 7},
            EdgeIndices{0, 4},
            EdgeIndices{1, 5},
            EdgeIndices{2, 6},
            EdgeIndices{3, 7},
        };

        // TODO: needs mesh_asset for wireframe
        inline Entity ConstructCube(EntityManager* manager);
        /*
        inline Entity ConstructCube() {
            Mesh mesh{ MeshData{}, MeshElements{1} };

            mesh.mesh_data.positions = utils::Buffer<VertexPosition>(positions);

            mesh.elements.elements[0] = utils::Buffer<uint32_t>(edges.size() * 2);
            std::memcpy(mesh.elements.elements[0].data.get(), edges.data(), sizeof(EdgeIndices) * edges.size());
            mesh.elements.element_types = utils::Buffer<ElementType>({ ElementType::Edge });

            Entity result;
            result.AddComponent(std::make_unique<Components::MeshRenderer>(std::move(mesh), std::make_unique<MaterialUnlit>(std::string{ "BaseMaterial" })));

            return result;
        }
        */

    } // namespace WireframeCube

    namespace ColoredCube {

        static constexpr std::initializer_list<VertexPosition> positions = {
            VertexPosition{-0.5f, 0.5f, 0.5f},
            VertexPosition{-0.5f, -0.5f, 0.5f},
            VertexPosition{0.5f, -0.5f, 0.5f},
            VertexPosition{0.5f, 0.5f, 0.5f},
            VertexPosition{-0.5f, 0.5f, -0.5f},
            VertexPosition{-0.5f, -0.5f, -0.5f},
            VertexPosition{0.5f, -0.5f, -0.5f},
            VertexPosition{0.5f, 0.5f, -0.5f},
        };

        static constexpr std::initializer_list<VertexColor> colors = {
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
        };

        static constexpr std::array triangles{
            TriangleIndices{0, 1, 3},
            TriangleIndices{1, 2, 3},
            TriangleIndices{3, 2, 7},
            TriangleIndices{2, 6, 7},
            TriangleIndices{6, 4, 7},
            TriangleIndices{6, 5, 4},
            TriangleIndices{5, 0, 4},
            TriangleIndices{5, 1, 0},
            TriangleIndices{1, 5, 2},
            TriangleIndices{2, 5, 6},
            TriangleIndices{0, 3, 4},
            TriangleIndices{4, 3, 7},
        };

        // TODO: needs mesh_asset for colored
        inline Entity ConstructCube(EntityManager* manager);
        /*
        inline Entity ConstructCube() {
            Mesh mesh{ MeshData{}, MeshElements{1} };

            mesh.mesh_data.positions = utils::Buffer<VertexPosition>(positions);
            mesh.mesh_data.colors = utils::Buffer<VertexColor>(colors);

            mesh.elements.elements[0] = utils::Buffer<uint32_t>(triangles.size() * 3);
            std::memcpy(mesh.elements.elements[0].data.get(), triangles.data(), sizeof(TriangleIndices) * triangles.size());
            mesh.elements.element_types = utils::Buffer<ElementType>({ ElementType::Triangle });

            Entity result;
            result.AddComponent(std::make_unique<Components::MeshRenderer>(std::move(mesh), std::make_unique<MaterialUnlit>(std::string{ "BaseMaterial" })));

            return result;
        }
        */

    } // namespace ColoredCube

    namespace TexturedCube {

        static constexpr std::initializer_list<VertexPosition> positions = {
            // Front
            VertexPosition{-0.5f, -0.5f, 0.5f},
            VertexPosition{0.5f, -0.5f, 0.5f},
            VertexPosition{-0.5f, 0.5f, 0.5f},
            VertexPosition{0.5f, 0.5f, 0.5f},
            // Right
            VertexPosition{0.5f, -0.5f, 0.5f},
            VertexPosition{0.5f, -0.5f, -0.5f},
            VertexPosition{0.5f, 0.5f, 0.5f},
            VertexPosition{0.5f, 0.5f, -0.5f},
            // Back
            VertexPosition{0.5f, -0.5f, -0.5f},
            VertexPosition{-0.5f, -0.5f, -0.5f},
            VertexPosition{0.5f, 0.5f, -0.5f},
            VertexPosition{-0.5f, 0.5f, -0.5f},
            // Left
            VertexPosition{-0.5f, -0.5f, -0.5f},
            VertexPosition{-0.5f, -0.5f, 0.5f},
            VertexPosition{-0.5f, 0.5f, -0.5f},
            VertexPosition{-0.5f, 0.5f, 0.5f},
            // Top
            VertexPosition{-0.5f, 0.5f, 0.5f},
            VertexPosition{0.5f, 0.5f, 0.5f},
            VertexPosition{-0.5f, 0.5f, -0.5f},
            VertexPosition{0.5f, 0.5f, -0.5f},
            // Bottom
            VertexPosition{-0.5f, -0.5f, -0.5f},
            VertexPosition{0.5f, -0.5f, -0.5f},
            VertexPosition{-0.5f, -0.5f, 0.5f},
            VertexPosition{0.5f, -0.5f, 0.5f},
        };

        static constexpr std::initializer_list<VertexColor> colors = {
            // Front
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            // Right
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
            // Back
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            // Left
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
            // Top
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
            // Bottom
            VertexColor{uint8_t{0}, uint8_t{255}, uint8_t{0}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{0}, uint8_t{0}, uint8_t{255}, uint8_t{255}},
            VertexColor{uint8_t{255}, uint8_t{0}, uint8_t{0}, uint8_t{255}},
        };

        static constexpr std::initializer_list<VertexUV> uvs = {
            // Front
            VertexUV{0.0f, 0.0f},
            VertexUV{1.0f, 0.0f},
            VertexUV{0.0f, 1.0f},
            VertexUV{1.0f, 1.0f},
            // Right
            VertexUV{0.0f, 0.0f},
            VertexUV{1.0f, 0.0f},
            VertexUV{0.0f, 1.0f},
            VertexUV{1.0f, 1.0f},
            // Back
            VertexUV{0.0f, 0.0f},
            VertexUV{1.0f, 0.0f},
            VertexUV{0.0f, 1.0f},
            VertexUV{1.0f, 1.0f},
            // Left
            VertexUV{0.0f, 0.0f},
            VertexUV{1.0f, 0.0f},
            VertexUV{0.0f, 1.0f},
            VertexUV{1.0f, 1.0f},
            // Top
            VertexUV{0.0f, 0.0f},
            VertexUV{1.0f, 0.0f},
            VertexUV{0.0f, 1.0f},
            VertexUV{1.0f, 1.0f},
            // Bottom
            VertexUV{0.0f, 0.0f},
            VertexUV{1.0f, 0.0f},
            VertexUV{0.0f, 1.0f},
            VertexUV{1.0f, 1.0f},
        };

        static constexpr std::array triangles{
            // Front
            TriangleIndices{0, 1, 2},
            TriangleIndices{1, 3, 2},
            // Right
            TriangleIndices{4, 5, 6},
            TriangleIndices{5, 7, 6},
            // Back
            TriangleIndices{8, 9, 10},
            TriangleIndices{9, 11, 10},
            // Left
            TriangleIndices{12, 13, 14},
            TriangleIndices{13, 15, 14},
            // Left
            TriangleIndices{16, 17, 18},
            TriangleIndices{17, 19, 18},
            // Bottom
            TriangleIndices{20, 21, 22},
            TriangleIndices{21, 23, 22},
        };

        inline std::shared_ptr<Entity> ConstructCube(EntityManager* manager) {
            AssetRef<Mesh> mesh{DefaultAssets.GetID("CubeUV")};

            Unlit_MaterialParameters parameters;
            parameters.base_texture = TextureBinding{
                .texture_ref = AssetRef<Texture2D>{DefaultAssets.GetID("DefaultTexture")},
                .uv_index = 0,
                .filtering = TextureFiltering::Nearest,
                .wrapping = math::vec2{TextureWrapping::Repeat},
            };
            parameters.face_culling = FaceCulling::Back;
            auto material_asset = std::make_unique<MaterialUnlit>(std::move(parameters));
            auto material_ID = AssetManager::Get()->RegisterAsset(std::move(material_asset));

            auto result = manager->AddEntity();
            result->AddComponent(std::make_unique<Components::Transform>());
            result->AddComponent(
                std::make_unique<Components::MeshRenderer>(std::move(mesh), AssetRef<MaterialUnlit>{material_ID}));
            return result;
        }

        inline std::shared_ptr<Entity> ConstructTransparentCube(EntityManager* manager) {
            AssetRef<Mesh> mesh{DefaultAssets.GetID("CubeUV")};

            Unlit_MaterialParameters parameters;
            parameters.base_texture = TextureBinding{
                .texture_ref = AssetRef<Texture2D>{DefaultAssets.GetID("DefaultTexture")},
                .uv_index = 0,
                .filtering = TextureFiltering::Nearest,
                .wrapping = math::vec2{TextureWrapping::Repeat},
            };
            parameters.blend_mode = BlendMode::Transparent;
            parameters.base_color.a = 0.5f;

            auto material_asset = std::make_unique<MaterialUnlit>(std::move(parameters));
            auto material_ID = AssetManager::Get()->RegisterAsset(std::move(material_asset));

            auto result = manager->AddEntity();
            result->AddComponent(std::make_unique<Components::Transform>());
            result->AddComponent(
                std::make_unique<Components::MeshRenderer>(std::move(mesh), AssetRef<MaterialUnlit>{material_ID}));
            return result;
        }

    } // namespace TexturedCube

} // namespace Core::Engine::Assets
