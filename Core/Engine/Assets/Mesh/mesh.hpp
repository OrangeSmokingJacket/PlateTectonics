#pragma once

#include <Math/math_def.hpp>
#include <Utils/buffer_utils.hpp>
#include <Core/Engine/Assets/Base/base.hpp>
#include <memory>

namespace Core::Engine::Assets {

	// TODO?: different storage types, OpenGL supports more than just float and uint32

	enum class VertexAttribute : uint16_t {
		Position = 1,
		Normal = 2,
		Tangent = 4,
		Color = 8,
		UV0 = 16,
		UV1 = 32,
		UV2 = 64,
		UV3 = 128,
		UV4 = 256,
		UV5 = 512,
		UV6 = 1024,
		UV7 = 2048
	};

	size_t VertexAttributePosition(VertexAttribute attribute);

    // underlying value could be used for calculating size
	enum class ElementType : uint8_t {
		Point = 1,
		Edge = 2,
		Triangle = 3,
	};

	constexpr size_t VertexAttributesCount = 12;

	using VertexPosition = math::vec3<float>;
	using VertexNormal = math::vec3<float>;
	using VertexTangent = math::vec3<float>;
	using VertexColor = math::vec4<uint8_t>;
	using VertexUV = math::vec2<float>;

	struct PointIndices {
		static constexpr ElementType element_type = ElementType::Point;

		uint32_t index;
	};

	struct EdgeIndices {
		static constexpr ElementType element_type = ElementType::Edge;

		std::array<uint32_t, 2> indices;
	};

	struct TriangleIndices {
		static constexpr ElementType element_type = ElementType::Triangle;

		std::array<uint32_t, 3> indices;
	};

	struct MeshData {
		utils::Buffer<VertexPosition> positions;
		utils::Buffer<VertexNormal> normals;
		utils::Buffer<VertexTangent> tangents;
		utils::Buffer<VertexColor> colors;
		std::array<utils::Buffer<VertexUV>, 8> uvs;
	};

    struct SubMeshElements {
        uint32_t start_index;
        uint32_t count;
        ElementType element_type;
    };

    struct MeshElements {
        utils::Buffer<uint32_t> elements;
        utils::Buffer<SubMeshElements> sub_meshes;
    };

    class Mesh final : public BaseAsset<Mesh> {
    public:
        explicit Mesh(AssetID ID, MeshData&& data, MeshElements&& elements) noexcept;
        explicit Mesh(MeshData&& data, MeshElements&& elements) noexcept;
        static constexpr AssetType asset_type = AssetType::Mesh;

		MeshData mesh_data;
		MeshElements elements;
	};

}