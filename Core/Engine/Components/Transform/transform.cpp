#include "transform.hpp"

#include <Math/matrix_transform.hpp>
#include <Math/scalar_ops.hpp>

namespace Core::Engine::Components {

	math::mat4x4<float> Transform::ToWorldSpace() const noexcept {
		// T * R * S ordering, with reduced computations
		auto res = RotationMatrix();
		res[0] *= scale.x;
		res[1] *= scale.y;
		res[2] *= scale.z;
		res[3][0] = position.x;
		res[3][1] = position.y;
		res[3][2] = position.z;
		return res;
	}

	math::mat4x4<float> Transform::ToLocalSpace() const noexcept {
		// S^-1 * R^-1 * T^-1 ordering
	    auto local = InverseScaleMatrix() * InverseRotationMatrix() * InverseTranslationMatrix();
		if (parent_transform) {
		    return local * parent_transform->ToLocalSpace();
		} else {
		    return local;
		}
	}

    // TODO: add flag to avoid multiple recalculations
    void Transform::UpdateWorldMatrix() {
	    if (parent_transform) {
	        parent_transform->UpdateWorldMatrix();
	        world_matrix = parent_transform->world_matrix * ToWorldSpace();
	    } else {
	        world_matrix = ToWorldSpace();
	    }
	}

    Transform::Transform(Transform* parent) noexcept
	    : parent_transform(parent)
        , rotation(EulerAngles{}) {}

    math::mat4x4<float> Transform::TranslationMatrix() const noexcept {
	    return math::transform::translation_matrix(position);
	}

    math::mat4x4<float> Transform::InverseTranslationMatrix() const noexcept {
	    return math::transform::translation_matrix(-position);
	}

	math::mat4x4<float> Transform::RotationMatrix() const noexcept {
	    if (std::holds_alternative<EulerAngles>(rotation)) {
	        const auto& rot = std::get<EulerAngles>(rotation);
	        auto r_x = math::transform::rotation_matrix_x(math::to_radians(rot.rotation_degrees.x));
	        auto r_y = math::transform::rotation_matrix_y(math::to_radians(rot.rotation_degrees.y));
	        auto r_z = math::transform::rotation_matrix_z(math::to_radians(rot.rotation_degrees.z));

	        switch (rot.ordering) {
	        case EulerRotationOrdering::XYZ:
	            return r_z * r_y * r_x;
	        case EulerRotationOrdering::XZY:
	            return r_y * r_z * r_x;
	        case EulerRotationOrdering::YXZ:
	            return r_z * r_x * r_y;
	        case EulerRotationOrdering::YZX:
	            return r_x * r_z * r_y;
	        case EulerRotationOrdering::ZXY:
	            return r_y * r_x * r_z;
	        case EulerRotationOrdering::ZYX:
	            return r_x * r_y * r_z;
	        default:
	            assert(false && "invalid rotation_ordering in transform");
	            return math::mat4x4<float>::identity();
	        }
	    } else {
	        return math::transform::rotation_matrix(std::get<math::quaternion<float>>(rotation));
	    }
	}

	math::mat4x4<float> Transform::InverseRotationMatrix() const noexcept {
	    if (std::holds_alternative<EulerAngles>(rotation)) {
	        const auto& rot = std::get<EulerAngles>(rotation);
	        auto r_x = math::transform::rotation_matrix_x(math::to_radians(-rot.rotation_degrees.x));
	        auto r_y = math::transform::rotation_matrix_y(math::to_radians(-rot.rotation_degrees.y));
	        auto r_z = math::transform::rotation_matrix_z(math::to_radians(-rot.rotation_degrees.z));

	        switch (InverseOrder(rot.ordering)) {
	        case EulerRotationOrdering::XYZ:
	            return r_z * r_y * r_x;
	        case EulerRotationOrdering::XZY:
	            return r_y * r_z * r_x;
	        case EulerRotationOrdering::YXZ:
	            return r_z * r_x * r_y;
	        case EulerRotationOrdering::YZX:
	            return r_x * r_z * r_y;
	        case EulerRotationOrdering::ZXY:
	            return r_y * r_x * r_z;
	        case EulerRotationOrdering::ZYX:
	            return r_x * r_y * r_z;
	        default:
	            assert(false && "invalid rotation_ordering in transform");
	            return math::mat4x4<float>::identity();
	        }
	    } else {
	        return math::transform::rotation_matrix(std::get<math::quaternion<float>>(rotation).inverse());
	    }
	}

    math::mat4x4<float> Transform::ScaleMatrix() const noexcept {
	    return math::transform::scaling_matrix(scale);
	}

    math::mat4x4<float> Transform::InverseScaleMatrix() const noexcept {
	    return math::transform::scaling_matrix(1.0f / scale);
	}
}
