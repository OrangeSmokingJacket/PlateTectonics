#pragma once

#include <Core/Engine/Assets/Material/material.hpp>
#include <Core/Engine/Assets/Mesh/mesh.hpp>
#include <Core/Engine/Components/Base/base.hpp>

namespace Core::Engine::Components {

    class MeshRenderer final : public BaseComponent {
    public:
        explicit MeshRenderer(Assets::AssetRef<Assets::Mesh>&& mesh,
                              Assets::AssetRef<Assets::MaterialUnlit>&& material);

        const Assets::AssetRef<Assets::Mesh>& MeshRef() const noexcept;
        const Assets::AssetRef<Assets::MaterialUnlit>& MaterialRef() const;

    private:
        Assets::AssetRef<Assets::Mesh> mesh_;
        utils::Buffer<Assets::AssetRef<Assets::MaterialUnlit>> materials_;
    };

} // namespace Core::Engine::Components
