#pragma once

#include <chrono>

namespace Core {

    class BaseLayer {
    public:
        BaseLayer() = default;
        BaseLayer(const BaseLayer&) = delete;
        BaseLayer(BaseLayer&&) = delete;
        BaseLayer& operator=(const BaseLayer&) = delete;
        BaseLayer& operator=(BaseLayer&&) = delete;

        virtual ~BaseLayer() = default;

        virtual void Update([[maybe_unused]] float delta_time) {}
        virtual void Render() {}
    };

}