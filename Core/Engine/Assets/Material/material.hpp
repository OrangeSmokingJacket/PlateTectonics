#pragma once

#include "../Texture/colors.hpp"
#include <Core/Engine/Assets/Base/base.hpp>
#include <Core/Engine/Assets/Texture/texture.hpp>
#include <Core/Engine/Assets/AssetRef/asset_ref.hpp>

#include <optional>
#include <variant>


namespace Core::Engine::Assets {

	enum class TextureWrapping : uint8_t {
		Repeat = 0,
		MirroredRepeat,
		ClampToEdge,
		ClampToBorder
	};

	enum class TextureFiltering : uint8_t {
		Nearest = 0,
		Linear,
	};

    enum class FaceCulling : uint8_t {
        None,
        Back,
        Front,
        Both,
    };

    enum class BlendMode : uint8_t {
        Opaque,
        AlphaClipping,
        Transparent,
        PremultipliedAlpha,
        Add,
        AddWeighted,
        Multiply,
        Multiply2,
        Screen,
        Subtract,
        Darken,
        Lighten,
    };

    constexpr bool IsOpaque(BlendMode mode) {
        return mode == BlendMode::Opaque || mode == BlendMode::AlphaClipping;
    }

    template<typename TextureClass>
	struct TextureBinding {
	    AssetRef<TextureClass> texture_ref; // none by default, since ID is InvalidID
        uint8_t uv_index = 0; // which MeshData UV is used for mapping
        // OpenGL does not natively support filtering for different axis
        TextureFiltering filtering = TextureFiltering::Nearest;
        math::impl::vector<TextureWrapping, TextureClass::Dimensions> wrapping{TextureWrapping::Repeat};
        math::impl::vector<float, TextureClass::Dimensions> uv_tiling{1.0f};
        math::impl::vector<float, TextureClass::Dimensions> uv_offset{0.0f};
    };

    // TODO: variance bitmap
    // TODO: GPU classes for each one
    // TODO: remove variants
    struct Unlit_MaterialParameters {
		static constexpr AssetType asset_type = AssetType::MaterialUnlit;
        
        Color base_color = Colors::White;
        TextureBinding<Texture2D> base_texture; // none by default
        FaceCulling face_culling = FaceCulling::None;
        BlendMode blend_mode = BlendMode::Opaque;
        bool use_vertex_color = false;
        bool wireframe_mode = false;
        float alpha_clip_threshold = 0.5f; // used only when BlendMode::AlphaClipping is enabled
    };

    struct PBR_MaterialParameters {
		static constexpr AssetType asset_type = AssetType::MaterialPBR;

        Color base_color;
        TextureBinding<Texture2D> base_texture;
        std::variant<float, TextureBinding<Texture2D_Mono>> metalic;
        std::variant<float, TextureBinding<Texture2D_Mono>> roughness;
        std::variant<float, TextureBinding<Texture2D_Mono>> specular;
        TextureBinding<Texture2D> normal;
        struct {
            float scale;
            TextureBinding<Texture2D_Mono> texture;
        } height_map;
        TextureBinding<Texture2D_Mono> occlusion;
        std::variant<Color, TextureBinding<Texture2D>> emission;
        FaceCulling face_culling;
        BlendMode blend_mode;
        float alpha_clip_threshold;
    };

    struct RT_MaterialParameters;


    // TODO: couple shader_ID_ with compiled shaders

    template <typename Parameters>
	class Material final : public BaseAsset<Material<Parameters>> {
	public:
		static constexpr AssetType asset_type = Parameters::asset_type;

	    Material() = default;
	    explicit Material(AssetID ID, Parameters&& params);
	    explicit Material(Parameters&& params);

		const Parameters& GetParameters() const noexcept;

	private:
		Parameters parameters_;
	};

    template <typename Parameters>
    Material<Parameters>::Material(AssetID ID, Parameters&& params)
        : BaseAsset<Material>(ID)
        , parameters_(std::move(params)) {}

    template <typename Parameters>
    Material<Parameters>::Material(Parameters&& params)
        : parameters_(std::move(params)) {}

    template <typename Parameters>
    auto Material<Parameters>::GetParameters() const noexcept -> const Parameters& { return parameters_; }


    using MaterialUnlit = Material<Unlit_MaterialParameters>;
    VALIDATE_ASSET_CLASS(MaterialUnlit);

}
