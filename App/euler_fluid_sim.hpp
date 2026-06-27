#pragma once

#include <cmath>
#include <cstring>
#include <memory>

#include <Math/math_def.hpp>

#include "Core/Engine/Assets/AssetRef/asset_ref.hpp"
#include "Core/Engine/Assets/Texture/colors.hpp"
#include "Core/Engine/Assets/Texture/texture.hpp"

namespace Core {
    class RenderPipeline;
}

template<typename T, size_t X, size_t Y, size_t Z>
struct Array3D : public std::array<std::array<std::array<T, Z>, Y>, X>
{
	static constexpr size_t size_X = X;
	static constexpr size_t size_Y = Y;
	static constexpr size_t size_Z = Z;
	static constexpr size_t cardinality = X * Y * Z;
	using value_type = T;
	using flat_iterator       = T*;
	using flat_const_iterator = const T*;
	
	flat_iterator flat_begin() { return std::array<std::array<std::array<T, Z>, Y>, X>::data()->data()->data(); }
	flat_iterator flat_end() { return flat_begin() + cardinality; }
	flat_const_iterator flat_cbegin() const { return std::array<std::array<std::array<T, Z>, Y>, X>::data()->data()->data(); }
	flat_const_iterator flat_cend() const { return flat_cbegin() + cardinality; }
	T* flat_data() { return reinterpret_cast<T*>(this); }
	const T* flat_data() const { return reinterpret_cast<const T*>(this); }
	
	void SetToZero();
	void SetTo(T value);
	void CopyFrom(const Array3D& other);
	T DotProduct(const Array3D& other) const;
	T Length() const;
};

struct Dimensions
{
	uint64_t x;
	uint64_t y;
	uint64_t z;
	
	[[nodiscard]] constexpr size_t Cardinality() const { return x * y * z; }
};

template <typename T>
struct CoefficientValues
{
	T Adiag;
	T Aplusx;
	T Aplusy;
	T Aplusz;
	
	T Sum() const { return Adiag + Aplusx + Aplusy + Aplusz; }
};


class EulerFluidSim {
public:	
	enum class InterpolationType : uint8_t
	{
		Trilinear = 0,
		Cubic = 1
	};
	
	// Dimensions
	static constexpr Dimensions dimensions{32, 16, 10};
	static constexpr Dimensions velocitiesXDimensions{dimensions.x + 1, dimensions.y, dimensions.z};
	static constexpr Dimensions velocitiesYDimensions{dimensions.x, dimensions.y + 1, dimensions.z};
	static constexpr Dimensions velocitiesZDimensions{dimensions.x, dimensions.y, dimensions.z + 1};
	
	using GridValues = Array3D<double, dimensions.x, dimensions.y, dimensions.z>;
	using GridCoefficientValues = Array3D<CoefficientValues<double>, dimensions.x, dimensions.y, dimensions.z>;
	using GridXValues = Array3D<double, velocitiesXDimensions.x, velocitiesXDimensions.y, velocitiesXDimensions.z>;
	using GridYValues = Array3D<double, velocitiesYDimensions.x, velocitiesYDimensions.y, velocitiesYDimensions.z>;
	using GridZValues = Array3D<double, velocitiesZDimensions.x, velocitiesZDimensions.y, velocitiesZDimensions.z>;

    // Temp render ptr to draw grid
	EulerFluidSim(Core::RenderPipeline* renderer);
	void Update(float delta_time);
    void RenderDebug(Core::RenderPipeline* renderer) const;
    void SetupRenderTexture();
    const Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df>& RenderTexture() const;

	void DebugRandomizeBorderVelocities();
	
	[[nodiscard]] math::vec3<float> NormalizePosition(const math::vec3<float>& pos) const; // TODO: make constexpr
	[[nodiscard]] math::vec3<float> DenormalizePosition(const math::vec3<float>& nPos) const; // TODO: make constexpr
	[[nodiscard]] math::vec3<float> SampleVelocity(const math::vec3<float>& pos) const;
	
private:
	static constexpr bool IsSolidCell(size_t x, size_t y, size_t z);
	
	void SetInitialTemperature();
	void CalculateCoefficients();
	void CalculatePreconditioner();
	void CalculateDivergence();
	void ApplyCoefficients(GridValues& result, const GridValues& apply_to);
	void ApplyPreconditioner(GridValues& result, const GridValues& apply_to);
	void PCG();
	void ProjectVelocities();
	void UpdateVelocities();
	void AdvectVelocities();
	void AdvectTemperature();
	void ApplyBuoyancy();
	
	void DrawGrid(Core::RenderPipeline* renderer, bool persistent) const;
	void DrawBoundaryVelocities(Core::RenderPipeline* renderer, bool persistent) const;
	void DrawInterpolatedVelocities(Core::RenderPipeline* renderer, bool persistent) const;

public:
    enum class DrawSlice : uint8_t {
        None = 0,
        Temperature,
        Pressure,
        PressureDivergence,
    } draw_slice = DrawSlice::None;
    // Settings variables
    // Physics
    // TODO: Graph for density over temperature
    // For now it is linear
    // Units: 1/K
    double thermal_expansion_coefficient = 1.0e-5;
    double min_thermal_expansion_coefficient = 1.0e-7;
    double max_thermal_expansion_coefficient = 1.0e-3;
    // Velocities
    bool draw_boundary_velocities = false;
    bool draw_interpolated_velocities = true;
    int cell_subdivision = 1;
    int max_cell_subdivision = 5;
    // Temperature
    Core::Engine::Assets::Color min_temperature_color = Core::Engine::Assets::FromSColor(Core::Engine::Assets::SColor{12, 22, 188, 255});
    Core::Engine::Assets::Color max_temperature_color = Core::Engine::Assets::FromSColor(Core::Engine::Assets::SColor{255, 186, 0, 255});
    // Solver debug information
    // PCG iterations
    uint32_t last_PCG_iteration_count = 0;
    // Pressure
    Core::Engine::Assets::Color min_pressure_color = Core::Engine::Assets::Colors::Magenta;
    Core::Engine::Assets::Color max_pressure_color = Core::Engine::Assets::Colors::Cyan;
    // Divergence
    Core::Engine::Assets::Color min_divergence_color = Core::Engine::Assets::Colors::Magenta;
    Core::Engine::Assets::Color max_divergence_color = Core::Engine::Assets::Colors::Cyan;

private:

	// Calculation related data
	// per cell values are allocated on the heap
	std::unique_ptr<GridValues> cell_pressure_;
	std::unique_ptr<GridValues> cell_temperature_, cell_temperature_temp_;
	std::unique_ptr<GridXValues> velocities_x_, velocities_x_temp_;
	std::unique_ptr<GridYValues> velocities_y_, velocities_y_temp_;
	std::unique_ptr<GridZValues> velocities_z_, velocities_z_temp_;
	std::unique_ptr<GridCoefficientValues> coefficient_;
	std::unique_ptr<GridValues> preconditioner_;
	std::unique_ptr<GridValues> intermediate_;
	std::unique_ptr<GridValues> cell_pressure_divergence_;
	std::unique_ptr<GridValues> residual_;
	std::unique_ptr<GridValues> auxiliary_;
    std::unique_ptr<GridValues> search_;

    math::vec3<float> world_origin_offset_;

    // Actual values
    static constexpr float scale = 1.0f; // meters
    static constexpr float fixed_time_step = 1 / 100.0; // seconds
    static constexpr double density = 1000.0; // kg/m^3
    static constexpr double solver_tolerance = 1.0e-6;
    static constexpr uint32_t max_solver_iterations = 100;
    static constexpr InterpolationType interpolation_type = InterpolationType::Trilinear;

    double ambient_temperature_ = 2500; // K
    double core_temperature_ = 5000; // K
    double gravitational_acceleration_ = 9.81;

    // Draw
    Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df> render_texture_;
    bool drawGrid = true;
    bool boundaryOnly = false;
    Core::Engine::Assets::SColor grid_color_{180, 180, 180, 80};

    Core::Engine::Assets::SColor boundary_velocity_color_x_ = Core::Engine::Assets::SColors::Red;
    Core::Engine::Assets::SColor boundary_velocity_color_y_ = Core::Engine::Assets::SColors::Green;
    Core::Engine::Assets::SColor boundary_velocity_color_z_ = Core::Engine::Assets::SColors::Blue;
    Core::Engine::Assets::SColor interpolated_velocity_color_ = Core::Engine::Assets::SColors::Cyan;
};

template <typename T, size_t X, size_t Y, size_t Z>
void Array3D<T, X, Y, Z>::SetToZero()
{
	std::memset(this->flat_data(), 0, cardinality * sizeof(T));
}

template <typename T, size_t X, size_t Y, size_t Z>
void Array3D<T, X, Y, Z>::SetTo(T value)
{
	for (size_t i = 0; i < cardinality; i++)
	{
		*(this->flat_data() + i) = value;
	}
}

template <typename T, size_t X, size_t Y, size_t Z>
void Array3D<T, X, Y, Z>::CopyFrom(const Array3D& other)
{
	std::memcpy(this->flat_data(), other.flat_data(), cardinality * sizeof(T));
}

template <typename T, size_t X, size_t Y, size_t Z>
T Array3D<T, X, Y, Z>::DotProduct(const Array3D& other) const
{
	T dotproduct{};
	for (size_t i = 0; i < cardinality; i++)
	{
		dotproduct += *(this->flat_data() + i) * *(other.flat_data() + i);
	}
	return dotproduct;
}

template <typename T, size_t X, size_t Y, size_t Z>
T Array3D<T, X, Y, Z>::Length() const
{
	return std::sqrt(this->DotProduct(*this));
}
