#pragma once

#include <Utils/uuid.hpp>

#include <cassert>
#include <memory>
#include <filesystem>

namespace Core::Engine::Assets {

	using AssetID = utils::UUIDv4;
	static constexpr AssetID InvalidAssetID = utils::UUIDv4::InvalidUUID();

    inline static const std::filesystem::path metadata_extension{".meta"};

    //! For now value matters for loading
    // Assigned values are used by the asset_manager/loader
	enum class AssetType : uint32_t {
		Invalid = 0,
		Texture2D = 1,
		Texture2D_Mono,
		Texture2Df,
	    Texture3Df,
		Mesh = 4,
		MaterialUnlit,
		MaterialPBR,
		MaterialRT,
	};

    // TODO: Mesh should be an imported asset
    constexpr bool IsNative(AssetType type) {
        assert(type != AssetType::Invalid);
        switch (type) {
        case AssetType::Texture2D:
        case AssetType::Texture2Df:
            return false;
        default:
            return true;
        }
    }
    constexpr bool IsImported(AssetType type) {
        return !IsNative(type);
    }

	class Asset {
    public:
        explicit Asset(AssetID ID = InvalidAssetID) noexcept;
		virtual ~Asset() = default;
		virtual AssetType GetAssetType() const noexcept = 0;

		AssetID ID;
	};

    template <typename Derived>
	class BaseAsset : public Asset {
	public:
        BaseAsset() noexcept = default;
        explicit BaseAsset(AssetID ID) noexcept;
		AssetType GetAssetType() const noexcept override { return Derived::asset_type; }
	};

    template <typename Derived>
    BaseAsset<Derived>::BaseAsset(AssetID ID) noexcept
        : Asset(ID) {}

#define VALIDATE_ASSET_CLASS(ClassName) static_assert(ClassName::asset_type == AssetType::ClassName && "Class name does not match the AssetType");

#define LOAD_ASSET(AssetClass, buffer)  AssetClass::LoadAsset(buffer);

}