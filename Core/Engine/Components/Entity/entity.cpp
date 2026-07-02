#include "entity.hpp"

namespace Core::Engine {

    Entity::Entity(Entity&& other) noexcept
        : components_(std::move(other.components_)) {
        RestoreEntityLink();
    }

    Entity& Entity::operator=(Entity&& other) noexcept {
        components_ = std::move(other.components_);
        RestoreEntityLink();
        return *this;
    }

    void Entity::Destroy() { is_alive_ = false; }

    void Entity::RestoreEntityLink() {
        for (const auto& entry : components_) {
            entry.second->LinkEntity(this);
        }
    }
} // namespace Core::Engine
