#include "asset_loader.hpp"

#include <fstream>

namespace Core::Engine::Assets {

    AssetHeader ConstructHeader(const Asset* asset, uint32_t version) {
        return AssetHeader{.storage_version = version, .asset_type = asset->GetAssetType(), .ID = asset->ID};
    }

    AssetHeader ReadHeder(std::filesystem::path metadata_path) {
        AssetHeader header;
        std::ifstream file(metadata_path, std::ios::binary);
        if (!file.is_open()) {
            // TODO: return error
            assert(false);
        }
        cereal::BinaryInputArchive archive(file);

        archive(header);
        return header;
    }

    void WriteHeder(const AssetHeader& header, std::filesystem::path metadata_path) {
        std::ofstream file(metadata_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            // TODO: return error
            assert(false);
        }

        cereal::BinaryOutputArchive serializer(file);
        serializer(header);
    }

    std::unique_ptr<Asset> LoadAsset(AssetType type, const std::filesystem::path& source_path) {
        switch (type) {
            case AssetType::Mesh:
                return LoadMesh(source_path);
            case AssetType::MaterialUnlit:
                return LoadMaterialUnlit(source_path);
            case AssetType::Texture2D:
                return LoadTexture2D(source_path);
            case AssetType::Texture2Df:
                return LoadTexture2Df(source_path);
            default:
                assert(false && "unhandled AssetType in LoadImportedAsset");
                return nullptr;
        }
    }
} // namespace Core::Engine::Assets
