#include "asset_manager.hpp"
#include <Core/Engine/Assets/AssetLoader/asset_loader.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/types/atomic.hpp>
#include <cereal/types/string.hpp>

#include <cassert>
#include <fstream>
#include <mutex>
#include <ranges>

// temp for setup
#include <Core/Engine/Assets/Base/default_assets.hpp>

namespace Core::Engine::Assets {

    static const std::filesystem::path assets_root_dir = std::filesystem::path(ASSETS_DIR).make_preferred();

    template <typename Archive>
    void save(Archive& archive, const AssetManager::Entry& entry) {
        archive(entry.asset_type, entry.path.string());
    }

    template <typename Archive>
    void load(Archive& archive, AssetManager::Entry& entry) {
        std::string path_str;
        archive(entry.asset_type, path_str);
        entry.path = path_str;
    }

    AssetManager::Entry::Entry(AssetType type, const std::filesystem::path& path)
        : asset_type(type)
        , path(path) {}

    AssetManager::Entry::Entry(std::unique_ptr<Asset>&& asset, std::filesystem::path&& path)
        : asset_type(asset->GetAssetType())
        , asset_data(std::move(asset))
        , path(std::move(path)) {}

    AssetManager::Entry::Entry(Entry&& other) noexcept
        : asset_type(other.asset_type)
        , asset_data(std::move(other.asset_data))
        , path(std::move(other.path)) {}

    AssetManager::Entry& AssetManager::Entry::operator=(Entry&& other) noexcept {
        asset_type = other.asset_type;
        asset_data = std::move(other.asset_data);
        path = std::move(other.path);
        return *this;
    }

    bool AssetManager::Entry::InMemory() const noexcept { return path.empty(); }

    AssetManager::AssetManager(utils::DefaultThreadPool* thread_pool)
        : thread_pool_(thread_pool) {
        static std::once_flag static_setup_flag;
        std::call_once(static_setup_flag, [this]() { s_asset_manager_ = this; });
    }

    AssetManager::~AssetManager() { s_asset_manager_ = nullptr; }

    bool AssetManager::Exists(AssetID ID) const noexcept { return entries_.contains(ID); }

    const Asset* AssetManager::GetAsset(AssetID ID) {
        assert(Exists(ID));
        auto* entry = &entries_.find(ID)->second;
        if (entry->asset_data) {
            return entry->asset_data.get();
        } else {
            LoadAsset(entry);
            return entry->asset_data.get();
        }
    }

    AssetID AssetManager::RegisterAsset(std::unique_ptr<Asset>&& asset, std::filesystem::path path) {
        AssetID ID = utils::GenerateUUIDv4();
        assert(!entries_.contains(ID));
        entries_.emplace(ID, Entry{std::move(asset), std::move(path)});
        return ID;
    }

    void AssetManager::PreloadAsset(AssetID ID) {
        assert(Exists(ID));
        Entry* entry = &entries_.find(ID)->second;
        if (!entry->asset_data) {
            // There is no need to lock map, because it is stable on inserts/erases
            thread_pool_->Enqueue([entry]() { LoadAsset(entry); });
        }
    }

    void AssetManager::PreloadAsset(AssetID ID, std::function<void(const Asset*)>&& callback) {
        assert(Exists(ID));
        Entry* entry = &entries_.find(ID)->second;
        if (!entry->asset_data) {
            // There is no need to lock map, because it is stable on inserts/erases
            thread_pool_->Enqueue([entry, cb = std::move(callback)]() {
                LoadAsset(entry);
                cb(entry->asset_data.get());
            });
        }
    }

    void AssetManager::FreeAsset(AssetID ID) {
        assert(entries_.contains(ID));
        if (auto it = entries_.find(ID); it->second.InMemory()) {
            entries_.erase(it);
        } else {
            it->second.asset_data = nullptr;
        }
    }

    void AssetManager::RemoveAsset(AssetID ID) {
        assert(entries_.contains(ID));
        auto it = entries_.find(ID);
        entries_.erase(it);
    }

    void AssetManager::SaveAssetAs(AssetID ID, const std::filesystem::path& path) {
        assert(entries_.contains(ID));
        auto it = entries_.find(ID);
        assert(it->second.InMemory());
        it->second.path = path;
        // TODO: make async
        SaveAsset(it->second.asset_data.get(), assets_root_dir / path);
    }

    void AssetManager::LoadAssetsEntries() {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(assets_root_dir)) {
            // Metadata files are skipped, to avoid dealing with dangling files.
            if (entry.is_directory() || entry.path().extension() == metadata_extension) {
                continue;
            }
            // Register assets have to have 'metadata_extension' file next to them
            auto metadata_path = entry.path();
            metadata_path += metadata_extension;
            // Different method deals with importing assets, since some of them require additional parameters
            if (std::filesystem::exists(metadata_path)) {
                auto header = ReadHeder(metadata_path);
                assert(!entries_.contains(header.ID));
                entries_.emplace(header.ID, Entry{header.asset_type, entry.path()});
            }
        }
    }

    AssetManager* AssetManager::Get() { return s_asset_manager_; }

    void AssetManager::LoadAsset(Entry* entry) {
        assert(entry);
        assert(entry->asset_type != AssetType::Invalid);
        assert(!entry->InMemory());

        entry->asset_data = Assets::LoadAsset(entry->asset_type, assets_root_dir / entry->path);
    }

} // namespace Core::Engine::Assets
