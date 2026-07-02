#include "entity_manager.hpp"

#include <algorithm>

namespace Core::Engine {

    EntityPtr EntityManager::AddEntity() {
        auto entity = std::shared_ptr<Entity>(new Entity(next_id_++));
        add_queue_.push_back(entity);
        return entity;
    }

    void EntityManager::ApplyChanges() {
        // remove dead entities
        auto it = std::partition(entities_.begin(), entities_.end(), [](const EntityPtr& entity) {
            return entity->is_alive_;
        });
        entities_.erase(it, entities_.end());
        // add new ones
        entities_.insert(entities_.end(), add_queue_.begin(), add_queue_.end());
        add_queue_.clear();
    }

    EntityVector& EntityManager::Entities() noexcept { return entities_; }

    const EntityVector& EntityManager::Entities() const noexcept { return entities_; }

} // namespace Core::Engine
