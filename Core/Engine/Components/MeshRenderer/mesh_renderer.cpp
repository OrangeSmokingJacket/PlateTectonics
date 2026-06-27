#include "mesh_renderer.hpp"

namespace Core::Engine::Components {

	MeshRenderer::MeshRenderer(Assets::AssetRef<Assets::Mesh>&& mesh, Assets::AssetRef<Assets::MaterialUnlit>&& material)
		: mesh_(mesh)
		, materials_(utils::Buffer<Assets::AssetRef<Assets::MaterialUnlit>>(1)) {
		materials_[0] = material;
	}

	const Assets::AssetRef<Assets::Mesh>& MeshRenderer::MeshRef() const noexcept {
		return mesh_;
	}

	const Assets::AssetRef<Assets::MaterialUnlit>& MeshRenderer::MaterialRef() const {
		return materials_[0];
	}

}