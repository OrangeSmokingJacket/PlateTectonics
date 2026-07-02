#pragma once

#include <Core/Engine/Components/Base/base.hpp>

#include <cassert>
#include <memory>
#include <typeindex>
#include <unordered_map>

namespace Core::Engine {

    using EntityID = size_t;
    // TODO: add name
    // TODO: tag
    class Entity {
    private:
        friend class EntityManager;
        // Can be constructed only through EntityManager
        explicit Entity(EntityID ID)
            : ID_(ID) {}

    public:
        Entity(const Entity&) = delete;
        Entity& operator=(const Entity&) = delete;
        Entity(Entity&&) noexcept;
        Entity& operator=(Entity&&) noexcept;
        virtual ~Entity() = default;

        // void Enable();
        // void Disable();
        void Destroy();

        template <typename ComponentClass>
        requires(std::derived_from<ComponentClass, Components::BaseComponent>)
        void AddComponent(std::unique_ptr<ComponentClass>&& component);
        template <typename ComponentClass, typename... Args>
        requires(std::derived_from<ComponentClass, Components::BaseComponent>)
        void AddComponent(Args&&... args);

        template <typename ComponentClass>
        requires(std::derived_from<ComponentClass, Components::BaseComponent>)
        ComponentClass* GetComponent();
        template <typename ComponentClass>
        requires(std::derived_from<ComponentClass, Components::BaseComponent>)
        const ComponentClass* GetComponent() const;

    private:
        void RestoreEntityLink();
        std::unordered_map<std::type_index, std::unique_ptr<Components::BaseComponent>> components_;
        EntityID ID_;
        bool is_alive_ = true;
        bool is_active_ = true;
    };

    template <typename ComponentClass>
    requires(std::derived_from<ComponentClass, Components::BaseComponent>)
    void Entity::AddComponent(std::unique_ptr<ComponentClass>&& component) {
        assert(!components_.contains(std::type_index(typeid(ComponentClass))));
        component->entity_ = this;
        components_.emplace(std::type_index(typeid(ComponentClass)), std::move(component));
    }

    template <typename ComponentClass, typename... Args>
    requires(std::derived_from<ComponentClass, Components::BaseComponent>)
    void Entity::AddComponent(Args&&... args) {
        AddComponent(std::make_unique<ComponentClass>(std::forward<Args>(args)...));
    }

    template <typename ComponentClass>
    requires(std::derived_from<ComponentClass, Components::BaseComponent>)
    ComponentClass* Entity::GetComponent() {
        if (auto it = components_.find(std::type_index(typeid(ComponentClass))); it == components_.end()) {
            return nullptr;
        } else {
            return static_cast<ComponentClass*>(it->second.get());
        }
    }

    template <typename ComponentClass>
    requires(std::derived_from<ComponentClass, Components::BaseComponent>)
    const ComponentClass* Entity::GetComponent() const {
        if (auto it = components_.find(std::type_index(typeid(ComponentClass))); it == components_.end()) {
            return nullptr;
        } else {
            return static_cast<const ComponentClass*>(it->second.get());
        }
    }

} // namespace Core::Engine
