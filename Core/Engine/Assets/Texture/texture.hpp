#pragma once

#include "texel_data.hpp"

#include <Core/Engine/Assets/Base/base.hpp>
#include <Core/Filesystem/filesystem.hpp>
#include <Utils/buffer_utils.hpp>

#include <cassert>
#include <limits>

namespace Core::Engine::Assets {

    namespace impl {

        template <TexelType T, TexelFormat F, size_t Dimensions>
        struct TextureAssetType;

        template <>
        struct TextureAssetType<TexelType::uint8, TexelFormat::RGBA, 2> {
            static constexpr AssetType type = AssetType::Texture2D;
        };

        template <>
        struct TextureAssetType<TexelType::uint8, TexelFormat::Luminance, 2> {
            static constexpr AssetType type = AssetType::Texture2D_Mono;
        };

        template <>
        struct TextureAssetType<TexelType::float32, TexelFormat::RGBA, 2> {
            static constexpr AssetType type = AssetType::Texture2Df;
        };

        template <>
        struct TextureAssetType<TexelType::float32, TexelFormat::RGBA, 3> {
            static constexpr AssetType type = AssetType::Texture3Df;
        };

    } // namespace impl

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    // implicitly requires (TextureDimensions > 0 && TextureDimensions <=3) because of constructors
    class Texture final : public BaseAsset<Texture<T, F, TextureDimensions>> {
    public:
        static constexpr AssetType asset_type = impl::TextureAssetType<T, F, TextureDimensions>::type;
        static constexpr TexelType texel_type = T;
        static constexpr TexelFormat texel_format = F;
        static constexpr size_t Dimensions = TextureDimensions;
        using storage_type = impl::TexelData<T, F>;

        // Texture1D
        explicit Texture(AssetID ID, size_t x) requires(TextureDimensions == 1);
        // Texture2D
        explicit Texture(AssetID ID, size_t x, size_t y) requires(TextureDimensions == 2);
        // CubeMapTexture
        explicit Texture(AssetID ID, size_t x, size_t y) requires(TextureDimensions == 3);
        // Texture3D
        explicit Texture(size_t x, size_t y, size_t z) requires(TextureDimensions == 3);
        explicit Texture(AssetID ID, size_t x, size_t y, size_t z) requires(TextureDimensions == 3);

        // total texel count
        size_t Cardinality() const noexcept;

        size_t Width() const noexcept { return size_x_; }
        size_t Height() const noexcept { return size_y_; }
        size_t Depth() const noexcept { return size_z_; }

        auto At(size_t x) noexcept -> storage_type& requires(TextureDimensions == 1);
        auto At(size_t x) const noexcept -> const storage_type& requires(TextureDimensions == 1);

        auto At(size_t x, size_t y) noexcept -> storage_type& requires(TextureDimensions == 2);
        auto At(size_t x, size_t y) const noexcept -> const storage_type& requires(TextureDimensions == 2);

        auto At(size_t x, size_t y, size_t z) noexcept -> storage_type& requires(TextureDimensions == 3);
        auto At(size_t x, size_t y, size_t z) const noexcept -> const storage_type& requires(TextureDimensions == 3);

        auto Data() noexcept -> storage_type*;
        auto Data() const noexcept -> const storage_type*;

    private:
        // For now, it most GPU supports up to 2^14 size, and some fileformats are up to 2^16
        uint16_t size_x_ = 1;
        uint16_t size_y_ = 1;
        uint16_t size_z_ = 1;
        utils::Buffer<storage_type> buffer_;
    };

    using Texture2D_Mono = Texture<TexelType::uint8, TexelFormat::Luminance, 2>;
    VALIDATE_ASSET_CLASS(Texture2D_Mono);
    using Texture2D = Texture<TexelType::uint8, TexelFormat::RGBA, 2>;
    VALIDATE_ASSET_CLASS(Texture2D);
    using Texture2Df = Texture<TexelType::float32, TexelFormat::RGBA, 2>;
    VALIDATE_ASSET_CLASS(Texture2Df);
    using Texture3Df = Texture<TexelType::float32, TexelFormat::RGBA, 3>;
    VALIDATE_ASSET_CLASS(Texture3Df);

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    Texture<T, F, TextureDimensions>::Texture(AssetID ID, size_t x) requires(TextureDimensions == 1)
        : BaseAsset<Texture>(ID)
        , size_x_(x)
        , buffer_(Cardinality()) {
        assert(x < std::numeric_limits<uint16_t>::max());
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    Texture<T, F, TextureDimensions>::Texture(AssetID ID, size_t x, size_t y) requires(TextureDimensions == 2)
        : BaseAsset<Texture>(ID)
        , size_x_(x)
        , size_y_(y)
        , buffer_(Cardinality()) {
        assert(x < std::numeric_limits<uint16_t>::max());
        assert(y < std::numeric_limits<uint16_t>::max());
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    Texture<T, F, TextureDimensions>::Texture(AssetID ID, size_t x, size_t y) requires(TextureDimensions == 3)
        : BaseAsset<Texture>(ID)
        , size_x_(x)
        , size_y_(y)
        , size_z_(6) // for each side of the cube
        , buffer_(Cardinality()) {
        assert(x < std::numeric_limits<uint16_t>::max());
        assert(y < std::numeric_limits<uint16_t>::max());
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    Texture<T, F, TextureDimensions>::Texture(size_t x, size_t y, size_t z) requires(TextureDimensions == 3)
        : size_x_(x)
        , size_y_(y)
        , size_z_(z)
        , buffer_(Cardinality()) {
        assert(x < std::numeric_limits<uint16_t>::max());
        assert(y < std::numeric_limits<uint16_t>::max());
        assert(z < std::numeric_limits<uint16_t>::max());
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    Texture<T, F, TextureDimensions>::Texture(AssetID ID, size_t x, size_t y, size_t z) requires(TextureDimensions == 3)
        : BaseAsset<Texture>(ID)
        , size_x_(x)
        , size_y_(y)
        , size_z_(z)
        , buffer_(Cardinality()) {
        assert(x < std::numeric_limits<uint16_t>::max());
        assert(y < std::numeric_limits<uint16_t>::max());
        assert(z < std::numeric_limits<uint16_t>::max());
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    size_t Texture<T, F, TextureDimensions>::Cardinality() const noexcept {
        return static_cast<size_t>(size_x_) * static_cast<size_t>(size_y_) * static_cast<size_t>(size_z_);
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::At(size_t x) noexcept -> storage_type& requires(TextureDimensions == 1)
    {
        assert(x < size_x_);
        return buffer_.data[x];
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::At(size_t x) const noexcept
        -> const storage_type& requires(TextureDimensions == 1)
    {
        assert(x < size_x_);
        return buffer_.data[x];
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::At(size_t x, size_t y) noexcept
        -> storage_type& requires(TextureDimensions == 2)
    {
        assert(x < size_x_);
        assert(y < size_y_);
        return buffer_.data[size_x_ * y + x];
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::At(size_t x, size_t y) const noexcept
        -> const storage_type& requires(TextureDimensions == 2)
    {
        assert(x < size_x_);
        assert(y < size_y_);
        return buffer_.data[size_x_ * y + x];
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::At(size_t x, size_t y, size_t z) noexcept
        -> storage_type& requires(TextureDimensions == 3)
    {
        assert(x < size_x_);
        assert(y < size_y_);
        assert(z < size_z_);
        return buffer_.data[size_x_ * size_y_ * z + size_x_ * y + x];
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::At(size_t x, size_t y, size_t z) const noexcept
        -> const storage_type& requires(TextureDimensions == 3)
    {
        assert(x < size_x_);
        assert(y < size_y_);
        assert(z < size_z_);
        return buffer_.data[size_x_ * size_y_ * z + size_x_ * y + x];
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::Data() noexcept -> storage_type* {
        return buffer_.data.get();
    }

    template <TexelType T, TexelFormat F, size_t TextureDimensions>
    auto Texture<T, F, TextureDimensions>::Data() const noexcept -> const storage_type* {
        return buffer_.data.get();
    }

} // namespace Core::Engine::Assets
