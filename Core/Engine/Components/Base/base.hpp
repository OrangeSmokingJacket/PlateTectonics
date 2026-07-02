#pragma once

namespace Core::Engine {
    class Entity;
}

namespace Core::Engine::Components {

    class BaseComponent {
    public:
        virtual ~BaseComponent() = default;

        const Engine::Entity* GetEntity() const noexcept;

    protected:
        virtual void LinkEntity(Engine::Entity* entity);
        friend class Engine::Entity;
        Engine::Entity* entity_ = nullptr;
    };

} // namespace Core::Engine::Components
