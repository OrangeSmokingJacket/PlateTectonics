#pragma once

#include <Math/math_def.hpp>
#include <Utils/enum_utils.hpp>
#include <Core/Engine/Components/Base/base.hpp>

#include <variant>

namespace Core::Engine::Components {

	enum class EulerRotationOrdering : uint8_t
	{
		XYZ = 1,
		XZY = 2,
		YXZ = 4,
		ZXY = 8,
		YZX = 16,
		ZYX = 32
	};

	constexpr EulerRotationOrdering InverseOrder(EulerRotationOrdering ordering) noexcept {
		return utils::from_underlying_type<EulerRotationOrdering>(uint8_t{32} / utils::to_underlying_type(ordering));
	}

    struct EulerAngles {
		math::vec3<float> rotation_degrees{};
		EulerRotationOrdering ordering = EulerRotationOrdering::XYZ;
	};

	class Transform final : public BaseComponent {
	public:
	    // defaults to EulerAngles
	    explicit Transform(Transform* parent = nullptr) noexcept;
	    template<typename RotationStruct>
	    explicit Transform(std::type_identity<RotationStruct>, Transform* parent = nullptr) noexcept;

        [[nodiscard]] math::mat4x4<float> TranslationMatrix() const noexcept;
	    [[nodiscard]] math::mat4x4<float> InverseTranslationMatrix() const noexcept;
	    [[nodiscard]] math::mat4x4<float> RotationMatrix() const noexcept;
	    [[nodiscard]] math::mat4x4<float> InverseRotationMatrix() const noexcept;
	    [[nodiscard]] math::mat4x4<float> ScaleMatrix() const noexcept;
	    [[nodiscard]] math::mat4x4<float> InverseScaleMatrix() const noexcept;

		[[nodiscard]] math::mat4x4<float> ToWorldSpace() const noexcept;
		[[nodiscard]] math::mat4x4<float> ToLocalSpace() const noexcept;

	    void UpdateWorldMatrix();


	    Transform* parent_transform = nullptr;

		math::vec3<float> position{};
	    std::variant<EulerAngles, math::quaternion<float>> rotation;
		math::vec3<float> scale{ 1.0f, 1.0f, 1.0f };

		math::mat4x4<float> world_matrix = math::mat4x4<float>::identity();
	};

    template <typename RotationStruct>
    Transform::Transform(std::type_identity<RotationStruct>, Transform* parent) noexcept
        : parent_transform(parent)
        , rotation(RotationStruct{}) {}
}
