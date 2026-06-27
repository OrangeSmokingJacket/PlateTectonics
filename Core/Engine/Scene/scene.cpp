#include "scene.hpp"

namespace Core::Engine {

    EntityManager& Scene::GetEntityManager() noexcept {
        return entity_manager_;
    }

    const EntityManager& Scene::GetEntityManager() const noexcept {
        return entity_manager_;
    }

    const Components::Camera* Scene::MainCamera() const noexcept {
        for (auto& entity : entity_manager_.Entities()) {
            if (auto camera = entity->GetComponent<Components::Camera>()) {
                return camera;
            }
        }
        return nullptr;
    }

}
