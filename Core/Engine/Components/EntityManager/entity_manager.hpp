#pragma once

#include <Core/Engine/Components/Entity/entity.hpp>

#include <vector>

namespace Core::Engine {

    using EntityPtr = std::shared_ptr<Entity>;
    using EntityVector = std::vector<EntityPtr>;
    class EntityManager final {
    public:
        explicit EntityManager() = default;

        // Entity would be created on the next frame
        EntityPtr AddEntity();
        void ApplyChanges();
        EntityVector& Entities() noexcept;
        const EntityVector& Entities() const noexcept;

    private:
        EntityVector add_queue_;
        EntityVector entities_;
        size_t next_id_{0};
    };

}