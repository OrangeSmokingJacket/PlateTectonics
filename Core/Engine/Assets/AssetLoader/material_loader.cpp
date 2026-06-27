#include "asset_loader.hpp"

#include <Core/Engine/Assets/Material/material.hpp>
#include <Math/impl/serialization.hpp>

#include <cereal/types/string.hpp>

#include <fstream>

namespace Core::Engine::Assets {

    template <typename Archive>
    void save(Archive& archive, const TextureBinding<Texture2D>& binding) {
        archive(binding.texture_ref,
                binding.uv_index,
                binding.wrapping,
                binding.filtering,
                binding.uv_tiling,
                binding.uv_offset);
    }

    template <typename Archive>
    void load(Archive& archive, TextureBinding<Texture2D>& binding) {
        archive(binding.texture_ref,
                binding.uv_index,
                binding.wrapping,
                binding.filtering,
                binding.uv_tiling,
                binding.uv_offset);
    }

    template <typename Archive>
    void save(Archive& archive, const Unlit_MaterialParameters& parameters) {
        archive(parameters.base_color,
                parameters.base_texture,
                parameters.face_culling,
                parameters.blend_mode,
                parameters.use_vertex_color,
                parameters.wireframe_mode,
                parameters.alpha_clip_threshold);
    }

    template <typename Archive>
    void load(Archive& archive, Unlit_MaterialParameters& parameters) {
        archive(parameters.base_color,
                parameters.base_texture,
                parameters.face_culling,
                parameters.blend_mode,
                parameters.use_vertex_color,
                parameters.wireframe_mode,
                parameters.alpha_clip_threshold);
    }

    std::unique_ptr<Asset> LoadMaterialUnlit(const std::filesystem::path& source_path) {
        auto metadata_path = source_path;
        metadata_path += metadata_extension;
        AssetHeader header = ReadHeder(metadata_path);
        assert(header.asset_type == AssetType::MaterialUnlit);

        Unlit_MaterialParameters parameters;
        std::ifstream file(source_path, std::ios::binary);
        if (!file.is_open()) {
            // TODO: return error
            assert(false);
        }
        cereal::BinaryInputArchive archive(file);

        archive(parameters);

        return std::make_unique<MaterialUnlit>(header.ID, std::move(parameters));
    }

    template <>
    void SaveAsset<MaterialUnlit>(const MaterialUnlit* asset, const std::filesystem::path& source_path) {
        auto metadata_path = source_path;
        metadata_path += metadata_extension;
        WriteHeder(ConstructHeader(asset), metadata_path);

        std::ofstream file(source_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            // TODO: return error
            assert(false);
        }

        cereal::BinaryOutputArchive serializer(file);
        serializer(asset->GetParameters());
    }
}
