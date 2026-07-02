#include "mesh.hpp"

namespace Core::Engine::Assets {
    size_t VertexAttributePosition(VertexAttribute attribute) {
        switch (attribute) {
            case VertexAttribute::Position:
                return 0;
            case VertexAttribute::Normal:
                return 1;
            case VertexAttribute::Tangent:
                return 2;
            case VertexAttribute::Color:
                return 3;
            case VertexAttribute::UV0:
                return 4;
            case VertexAttribute::UV1:
                return 5;
            case VertexAttribute::UV2:
                return 6;
            case VertexAttribute::UV3:
                return 7;
            case VertexAttribute::UV4:
                return 8;
            case VertexAttribute::UV5:
                return 9;
            case VertexAttribute::UV6:
                return 10;
            case VertexAttribute::UV7:
                return 11;
            default:
                assert(false);
                return size_t(-1);
        }
    }

    Mesh::Mesh(AssetID ID, MeshData&& data, MeshElements&& elements) noexcept
        : BaseAsset(ID)
        , mesh_data(std::move(data))
        , elements(std::move(elements)) {}

    Mesh::Mesh(MeshData&& data, MeshElements&& elements) noexcept
        : mesh_data(std::move(data))
        , elements(std::move(elements)) {}

} // namespace Core::Engine::Assets
