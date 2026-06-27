#pragma once

#include <Core/Engine/Assets/Base/base.hpp>
#include <Core/Engine/Assets/AssetManager/asset_manager.hpp>

namespace Core::Engine::Assets {

    // Optionally holds reference to an asset
    // ID == InvalidAssetID -> no asset ref
    template<typename T>
    struct AssetRef {
        AssetRef(AssetID ID = InvalidAssetID) : ID(ID) {}

        bool HasAsset() const noexcept;
        const T* GetAsset() const;

        AssetID ID;
    };

    template <typename T, typename Archive>
    void save(Archive& archive, const AssetRef<T>& ref) {
        archive(ref.ID);
    }

    template <typename T, typename Archive>
    void load(Archive& archive, AssetRef<T>& ref) {
        archive(ref.ID);
    }

    template <typename T>
    bool AssetRef<T>::HasAsset() const noexcept { return ID != InvalidAssetID; }

    template <typename T>
    const T* AssetRef<T>::GetAsset() const {
        if (HasAsset()) {
            return static_cast<const T*>(AssetManager::Get()->GetAsset(ID));
        } else {
            return nullptr;
        }
    }
}
