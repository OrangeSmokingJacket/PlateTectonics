#include "camera.hpp"

#include <math/matrix_transform.hpp>

template <typename T>
math::vec3<T> rotate_vector(T rads, const math::vec3<T>& vec) {
    return static_cast<math::vec3<T>>(math::transform::rotation_matrix_y(rads) * math::vec4<T>(vec, T{1}));
}

namespace Core::Engine::Components {

    Camera::Camera(float FOV, float near_clip, float far_clip)
        : FOV_(FOV)
        , near_clip_(near_clip)
        , far_clip_(far_clip) {}

    float Camera::FOV() const noexcept { return FOV_; }

    math::mat4x4<float> Camera::ViewMatrix(const Transform* transfrom) const noexcept {
        auto local = transfrom->InverseRotationMatrix() * transfrom->InverseTranslationMatrix();
        if (transfrom->parent_transform) {
            return local * transfrom->parent_transform->ToLocalSpace();
        } else {
            return local;
        }
    }

    math::mat4x4<float> Camera::PerspectiveMatrix(float aspect_ratio) const noexcept {
        return math::transform::perspective_matrix(FOV_, aspect_ratio, near_clip_, far_clip_);
    }

} // namespace Core::Engine::Components
