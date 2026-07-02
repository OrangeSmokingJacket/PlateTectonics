#pragma once
#include <math/math_def.hpp>

#include <Core/Engine/Components/Base/base.hpp>
#include <Core/Engine/Components/Transform/transform.hpp>

namespace Core::Engine::Components {

    class Camera final : public BaseComponent {
    public:
        explicit Camera(float FOV_rads, float near_clip, float far_clip);

        float FOV() const noexcept;
        [[nodiscard]] math::mat4x4<float> ViewMatrix(const Transform* transfrom) const noexcept;
        [[nodiscard]] math::mat4x4<float> PerspectiveMatrix(float aspect_ratio) const noexcept;

    private:
        // TODO: group into settings struct
        float FOV_;
        float near_clip_;
        float far_clip_;
        // TODO: render target
        // TODO: different perspectives
    };

} // namespace Core::Engine::Components
