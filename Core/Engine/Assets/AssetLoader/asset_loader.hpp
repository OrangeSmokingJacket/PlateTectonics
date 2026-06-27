#pragma once

#include <Core/Engine/Assets/Base/base.hpp>
#include <Core/Engine/Assets/Material/material.hpp>


namespace Core::Engine::Assets {

    // Metadata attached to all Assets
    struct AssetHeader {
        // could be much less than 32 bits, but there is a padding anyway
        uint32_t storage_version = 0;
        AssetType asset_type = AssetType::Invalid;
        AssetID ID = InvalidAssetID;
    };

    template <typename Archive>
    void save(Archive& archive, const AssetHeader& header) {
        archive(header.storage_version, header.asset_type, header.ID);
    }

    template <typename Archive>
    void load(Archive& archive, AssetHeader& header) {
        archive(header.storage_version, header.asset_type, header.ID);
    }

    AssetHeader ConstructHeader(const Asset* asset, uint32_t version = 0);
    AssetHeader ReadHeder(std::filesystem::path metadata_path);
    void WriteHeder(const AssetHeader& header, std::filesystem::path metadata_path);

    class Mesh;

    // TODO: profile direct reads from file and loading into a buffer
    std::unique_ptr<Asset> LoadMesh(const std::filesystem::path& source_path);
    std::unique_ptr<Asset> LoadMaterialUnlit(const std::filesystem::path& source_path);
    std::unique_ptr<Asset> LoadMaterialPBR(const std::filesystem::path& source_path);
    std::unique_ptr<Asset> LoadMaterialRT(const std::filesystem::path& source_path);
    std::unique_ptr<Asset> LoadTexture2D(const std::filesystem::path& source_path);
    std::unique_ptr<Asset> LoadTexture2Df(const std::filesystem::path& source_path);

    // Default one only saves metadata
    template<typename T>
    void SaveAsset(const T* asset, const std::filesystem::path& source_path) {
        auto metadata_path = source_path;
        metadata_path += metadata_extension;
        WriteHeder(ConstructHeader(asset), metadata_path);
    }

    template<>
    void SaveAsset(const Mesh* asset, const std::filesystem::path& source_path);
    template<>
    void SaveAsset(const MaterialUnlit* asset, const std::filesystem::path& source_path);
    /*
    template<>
    void SaveAsset(const MaterialPBR* asset, const std::filesystem::path& source_path);
    template<>
    void SaveAsset(const MaterialRT* asset, const std::filesystem::path& source_path);
    // For now textures use default one
    template<>
    void SaveAsset(const Texture2D* asset, const std::filesystem::path& source_path, const std::filesystem::path& metadata_path);
    template<>
    void SaveAsset(const Texture2Df* asset, const std::filesystem::path& source_path, const std::filesystem::path& metadata_path);
    */

    // TODO: load AssetType from metadata file
    std::unique_ptr<Asset> LoadAsset(AssetType type, const std::filesystem::path& source_path);

}
