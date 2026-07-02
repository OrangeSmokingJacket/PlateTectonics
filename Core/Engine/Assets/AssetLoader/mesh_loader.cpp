#include <fstream>

#include "asset_loader.hpp"

#include <Core/Engine/Assets/Mesh/mesh.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/types/array.hpp>

#include <Core/Engine/Components/EntityManager/basic_entities.hpp>

namespace Core::Engine::Assets {

    template <typename Archive>
    void save(Archive& archive, const MeshData& mesh_data) {
        archive(mesh_data.positions, mesh_data.normals, mesh_data.tangents, mesh_data.colors, mesh_data.uvs);
    }

    template <typename Archive>
    void load(Archive& archive, MeshData& mesh_data) {
        archive(mesh_data.positions, mesh_data.normals, mesh_data.tangents, mesh_data.colors, mesh_data.uvs);
    }

    template <typename Archive>
    void save(Archive& archive, const SubMeshElements& sub_mesh) {
        archive(sub_mesh.start_index, sub_mesh.count, sub_mesh.element_type);
    }

    template <typename Archive>
    void load(Archive& archive, SubMeshElements& sub_mesh) {
        archive(sub_mesh.start_index, sub_mesh.count, sub_mesh.element_type);
    }

    template <typename Archive>
    void save(Archive& archive, const MeshElements& mesh_elements) {
        archive(mesh_elements.elements, mesh_elements.sub_meshes);
    }

    template <typename Archive>
    void load(Archive& archive, MeshElements& mesh_elements) {
        archive(mesh_elements.elements, mesh_elements.sub_meshes);
    }

    std::unique_ptr<Asset> LoadMesh(const std::filesystem::path& source_path) {
        auto metadata_path = source_path;
        metadata_path += metadata_extension;
        AssetHeader header = ReadHeder(metadata_path);
        assert(header.asset_type == AssetType::Mesh);

        MeshData mesh_data;
        MeshElements mesh_elements;
        std::ifstream file(source_path, std::ios::binary);
        if (!file.is_open()) {
            // TODO: return error
            assert(false);
        }
        cereal::BinaryInputArchive archive(file);

        archive(mesh_data, mesh_elements);

        return std::make_unique<Mesh>(header.ID, std::move(mesh_data), std::move(mesh_elements));
    }

    template <>
    void SaveAsset<Mesh>(const Mesh* asset, const std::filesystem::path& source_path) {
        auto metadata_path = source_path;
        metadata_path += metadata_extension;
        WriteHeder(ConstructHeader(asset), metadata_path);

        std::ofstream file(source_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            // TODO: return error
            assert(false);
        }

        cereal::BinaryOutputArchive serializer(file);
        serializer(asset->mesh_data, asset->elements);
    }

} // namespace Core::Engine::Assets
