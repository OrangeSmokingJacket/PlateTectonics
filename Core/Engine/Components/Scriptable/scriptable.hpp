#pragma once

#include <Core/Engine/Components/Base/base.hpp>

#include <memory>
#include <vector>

namespace Core::Engine::Components {

    class Scriptable : public BaseComponent {
    public:
        virtual void Start() {}
        virtual void Update([[maybe_unused]] float delta_time) {}

        friend class ScriptablePack;
    };

} // namespace Core::Engine::Components
