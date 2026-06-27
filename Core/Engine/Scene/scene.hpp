#pragma once

#include <Core/Engine/Components/Camera/camera.hpp>
#include <Core/Engine/Components/EntityManager/entity_manager.hpp>

namespace Core::Engine {

    class Scene {
    public:
        EntityManager& GetEntityManager() noexcept;
        const EntityManager& GetEntityManager() const noexcept;

        const Components::Camera* MainCamera() const noexcept;

    private:
        EntityManager entity_manager_;
    };

}
