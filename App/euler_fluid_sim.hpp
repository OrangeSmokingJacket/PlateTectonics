#pragma once

#include "simulation_domain.hpp"

#include <Core/Engine/Assets/AssetRef/asset_ref.hpp>
#include <Core/Engine/Assets/Texture/colors.hpp>
#include <Core/Engine/Assets/Texture/texture.hpp>
#include <Math/math_def.hpp>
#include <Utils/thread_pool.hpp>

#include <cmath>
#include <memory>

namespace Core {
    class RenderPipeline;
}

namespace SimulationConstants {
    static constexpr float scale = 1.0f;                // meters
    static constexpr float fixed_time_step = 1 / 100.0; // seconds
    static constexpr double density = 1000.0;           // kg/m^3
    static constexpr double solver_tolerance = 1.0e-6;
    static constexpr uint32_t max_solver_iterations = 100;
    static constexpr InterpolationType interpolation_type = InterpolationType::Trilinear;
} // namespace SimulationConstants

// Solver is based upon FLUID SIMULATION SIGGRAPH 2007 Course Notes
class EulerFluidSim {
    friend SimulationDomain;

public:
    // Dimensions
    static constexpr Dimensions domains{2, 2, 2};
    static constexpr Dimensions full_dimensions =
        SimulationDomain::stripped_dimensions * domains + 2ull * SimulationDomain::halo_size;

    // Temp render ptr to draw grid
    explicit EulerFluidSim(Core::RenderPipeline* renderer, utils::DefaultThreadPool* thread_pool);
    void Update(float delta_time);
    void RenderDebug(Core::RenderPipeline* renderer) const;
    void SetupRenderTexture();
    const Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df>& RenderTexture() const;

    void DebugRandomizeBorderVelocities();

private:
    static constexpr bool IsSolidCell(const Index3D& world_index) {
        return world_index.x == 0 || world_index.y == 0 || world_index.z == 0 ||
               world_index.x + 1 >= full_dimensions.x || world_index.y + 1 >= full_dimensions.y ||
               world_index.z + 1 >= full_dimensions.z;
    }
    void SyncVelocitiesX();
    void SyncVelocitiesY();
    void SyncVelocitiesZ();
    void SyncVelocities();
    void SyncGridValues(std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* grid);
    void SyncCoefficients();

    void SetInitialTemperature();
    void PCG();
    void ProjectVelocities();

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
    int max_cell_subdivision = 4;
    // Temperature
    Core::Engine::Assets::Color min_temperature_color =
        Core::Engine::Assets::FromSColor(Core::Engine::Assets::SColor{12, 22, 188, 255});
    Core::Engine::Assets::Color max_temperature_color =
        Core::Engine::Assets::FromSColor(Core::Engine::Assets::SColor{255, 186, 0, 255});
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
    utils::DefaultThreadPool* thread_pool_;
    // Calculation related data
    // per cell values are allocated on the heap
    std::unique_ptr<std::array<SimulationDomain, Cardinality(domains)>> domains_;

    math::vec3<float> world_origin_offset_;

    // Actual values
    double crust_temperature_ = 0;      // K
    double ambient_temperature_ = 2500; // K
    double core_temperature_ = 5000;    // K
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

template <size_t AxisIndex>
constexpr size_t WorldIndex(size_t domain_index, size_t cell_index) {
    return domain_index * (SimulationDomain::dimensions[AxisIndex] - 2 * SimulationDomain::halo_size[AxisIndex]) +
           cell_index;
}

constexpr Index3D WorldIndex(const Index3D& domain_index, const Index3D& cell_index) {
    return Index3D{WorldIndex<0>(domain_index[0], cell_index[0]),
                   WorldIndex<1>(domain_index[1], cell_index[1]),
                   WorldIndex<2>(domain_index[2], cell_index[2])};
}
