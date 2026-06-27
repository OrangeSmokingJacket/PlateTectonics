#include "base.hpp"

namespace Core::Engine::Components {

    const Engine::Entity* BaseComponent::GetEntity() const noexcept {
        return entity_;
    }

    void BaseComponent::LinkEntity(Engine::Entity* entity) {
        entity_ = entity;
    }

}
