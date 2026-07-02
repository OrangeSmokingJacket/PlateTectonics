#pragma once

#include <Core/Engine/Assets/Base/base.hpp>
#include <Utils/buffer_utils.hpp>
#include <Utils/thread_pool.hpp>

#include <atomic>
#include <cstring>
#include <functional>
#include <filesystem>
// TODO: use flat map/set
#include <unordered_map>

namespace Core::Engine::Assets {

	class AssetManager {
	public:
		struct Entry {
			AssetType asset_type{AssetType::Invalid};
			std::unique_ptr<Asset> asset_data{nullptr};
		    // local path from 'Assets' root
			std::filesystem::path path{""};

		    Entry() = default;
		    explicit Entry(AssetType type, const std::filesystem::path& path);
		    explicit Entry(std::unique_ptr<Asset>&& asset, std::filesystem::path&& path);
		    Entry(Entry&& other) noexcept;
		    Entry& operator=(Entry&& other) noexcept;

		    bool InMemory() const noexcept;
		};

		struct EntriesPack {
			size_t pack_size;
			utils::Buffer<AssetType> asset_types;
		    utils::Buffer<std::unique_ptr<Asset>> assets;
		    // local path from 'Assets' root
		    std::filesystem::path path{""};

		    bool InMemory() const noexcept;
		};

	    explicit AssetManager(utils::DefaultThreadPool* thread_pool);
	    ~AssetManager();

		// Check if AssetManager has that ID
		bool Exists(AssetID ID) const noexcept;
		// Returns ptr to a LOADED asset
		// May invoke disk I/O to get it
		[[nodiscard]] const Asset* GetAsset(AssetID ID);
	    // Generates AssetID, verifies that it is unique
	    // path -> local to Assets directory
	    // If path is empty -> registers Asset as 'in-memory'
        [[nodiscard]] AssetID RegisterAsset(std::unique_ptr<Asset>&& asset, std::filesystem::path path = {});

		// Asynchronously loads asset in memory
	    // TODO: add variant returning a future
		void PreloadAsset(AssetID ID);
		// Asynchronously loads asset in memory and invokes callback
	    // Call back will be called from worker thread
		void PreloadAsset(AssetID ID, std::function<void(const Asset*)>&& callback);
		// Forcefully deallocates Asset data for Entry
	    // If Entry is 'in-memory' -> removes it entirely
		void FreeAsset(AssetID ID);
		void RemoveAsset(AssetID ID);
	    // For now works only for 'in-memory' Assets
	    // Takes on local path to 'Assets' directory
	    void SaveAssetAs(AssetID ID, const std::filesystem::path& path);
		// Loads Entries for all registered assets in 'Assets' directory
		// Does not necessary loads assets into memory
	    // Does not invoke importing, if metadata file is missing
		void LoadAssetsEntries();
		// Loads new Entries for assets in 'Assets' directory
		// Assigns ID to them
		void ScanForNewAssets();
		// Convert to one of known Asset types, if possible
		// Registers AssetID
		AssetID ImportAsset(const std::filesystem::path& path);

	    static AssetManager* Get();

	private:
		static void LoadAsset(Entry* entry);

        inline static AssetManager* s_asset_manager_ = nullptr;

	    utils::DefaultThreadPool* thread_pool_;
		std::unordered_map<AssetID, Entry> entries_;
		// TODO: multiple AssetID to one pack
		//std::unordered_map<AssetID, EntriesPack> entries_;
	};

}