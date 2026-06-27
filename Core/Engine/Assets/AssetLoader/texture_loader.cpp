#include "asset_loader.hpp"

#include <stb/stb_image.h>
#include <Core/Engine/Assets/Texture/texture.hpp>

#include <fstream>

namespace Core::Engine::Assets {

    std::unique_ptr<Asset> LoadTexture2D(const std::filesystem::path& source_path) {
        auto metadata_path = source_path;
        metadata_path += metadata_extension;
        AssetHeader header = ReadHeder(metadata_path);
        assert(header.asset_type == AssetType::Texture2D);

        stbi_set_flip_vertically_on_load(true);
        int size_x, size_y, file_channels;
        auto* data = stbi_load(source_path.string().c_str(), &size_x, &size_y, &file_channels, 4);
        assert(data);
        auto result = std::make_unique<Texture2D>(std::move(header.ID), size_x, size_y);
        std::memcpy(result->Data(), data, result->Cardinality() * sizeof(Texture2D::storage_type));
        stbi_image_free(data);
        return result;
    }

    std::unique_ptr<Asset> LoadTexture2Df(const std::filesystem::path& source_path) {
        auto metadata_path = source_path;
        metadata_path += metadata_extension;
        AssetHeader header = ReadHeder(metadata_path);
        assert(header.asset_type == AssetType::Texture2Df);

        stbi_set_flip_vertically_on_load(true);
        int size_x, size_y, file_channels;
        auto* data = stbi_loadf(source_path.string().c_str(), &size_x, &size_y, &file_channels, 4);
        assert(data);
        auto result = std::make_unique<Texture2Df>(std::move(header.ID), size_x, size_y);
        std::memcpy(result->Data(), data, result->Cardinality() * sizeof(Texture2Df::storage_type));
        stbi_image_free(data);
        return result;
    }

}
