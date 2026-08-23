#pragma once

#include "simulation_domain.hpp"

#include <Core/Engine/Assets/AssetRef/asset_ref.hpp>
#include <Core/Engine/Assets/Texture/colors.hpp>
#include <Core/Engine/Assets/Texture/texture.hpp>
#include <Math/math_def.hpp>
#include <Utils/thread_pool.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <memory>

namespace Core {
    class RenderPipeline;
}

// TODO: Adjust simulation parameters closer to reality
// For now PCG fights numerical errors
namespace SimulationConstants {
    static constexpr float scale = 2.0f;                // meters
    static constexpr float fixed_time_step = 1 / 100.0; // seconds
    static constexpr double density = 1000.0;           // kg/m^3
    static constexpr double solver_tolerance = 1.0e-6;
    static constexpr uint32_t max_solver_iterations = 100;
    static constexpr InterpolationType interpolation_type = InterpolationType::Trilinear;
    static constexpr size_t domains = 6;
} // namespace SimulationConstants

// Not all operations read diagonal values, so they could be skipped
enum class SyncScope : uint8_t { Orthogonal, Full };

// assume that there is no padding needed between them
// sim_dom constraints guarantee that Elements % 32 == 0
// TODO: proper RAII
template <typename T, size_t Elements, size_t Segments>
struct SegmentedBuffer {
    using value_t = T;
    static constexpr size_t elements = Elements;
    static constexpr size_t segments = Segments;
    static constexpr size_t allocation_size = sizeof(T) * elements * segments;

    T* ptr;

    T* GetSegment(size_t index) const {
        assert(index < segments);
        return ptr + elements * index;
    }
};

// TODO: proper RAII
template <typename T, size_t Elements, size_t PadTo, size_t Segments>
struct SwapBuffer {
    using value_t = T;
    static constexpr size_t elements = Elements;
    static constexpr size_t elements_padded = ((Elements - 1) / PadTo + 1) * PadTo;
    static constexpr size_t segments = Segments;
    static constexpr size_t allocation_size = sizeof(T) * elements_padded * segments;

    T* b1;
    T* b2;
    T** front_buffer;
    T** back_buffer;

    T* GetFrontSegment(size_t index) const {
        assert(index < segments);
        return *front_buffer + elements_padded * index;
    }

    T* GetBackSegment(size_t index) const {
        assert(index < segments);
        return *back_buffer + elements_padded * index;
    }

    void SwapBuffers() noexcept { std::swap(b1, b2); }
};

struct CudaDispatchBuffers {
    static constexpr size_t padding_size = 16;
    template <typename T>
    using SB = SegmentedBuffer<T, Cardinality(SimulationDomain::dimensions), SimulationConstants::domains>;
    template <typename T>
    using VxSB =
        SwapBuffer<T, Cardinality(SimulationDomain::velocitiesXDimensions), padding_size, SimulationConstants::domains>;
    template <typename T>
    using VySB =
        SwapBuffer<T, Cardinality(SimulationDomain::velocitiesYDimensions), padding_size, SimulationConstants::domains>;
    template <typename T>
    using VzSB =
        SwapBuffer<T, Cardinality(SimulationDomain::velocitiesZDimensions), padding_size, SimulationConstants::domains>;

    double* reduction_buffer;
    SB<double> pressure;
    SB<double> divergence;
    SB<double> auxiliary;
    SB<double> residual;
    SB<double> search;
    SB<CoefficientsValues<double>> coefficients;
    SB<CoefficientsValues<double>> preconditioner;
    VxSB<double> velocities_x;
    VySB<double> velocities_y;
    VzSB<double> velocities_z;
};
struct CudaGraphs {
    cudaStream_t stream;

    cudaGraphExec_t calculate_divergence;
    cudaGraphExec_t sync_divergence;
    cudaGraphExec_t PCG_Solver;
};

class EulerFluidSim {
    friend SimulationDomain;

public:
    // 1 domain per face
    static constexpr size_t domains = SimulationConstants::domains;
    static constexpr math::vec3<float> world_origin_offset{
        -(SimulationDomain::real_dimensions[0] * SimulationConstants::scale * 0.5f),
        -(SimulationDomain::real_dimensions[1] * SimulationConstants::scale * 0.5f),
        // x and y are equal, but we have to pick one
        SimulationDomain::real_dimensions[0] * SimulationConstants::scale * 0.5f};
    static constexpr float inner_sphere_radius = world_origin_offset[2];
    static constexpr size_t total_cells = domains * Cardinality(SimulationDomain::dimensions);
    // max safe 2^N, where total_cells % (2^N) == 0
    static constexpr size_t max_pow2_multiple = 64;

    // Temp render ptr to draw grid
    explicit EulerFluidSim(Core::RenderPipeline* renderer, utils::DefaultThreadPool* thread_pool);
    ~EulerFluidSim();
    void Update(float delta_time);
    void RenderDebug(Core::RenderPipeline* renderer) const;
    void SetupRenderTexture();
    const std::array<Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df>, 6>& RenderTextures() const;

    void DebugRandomizeBorderVelocities();

private:
    static constexpr bool IsSolidCell(size_t world_index_z) {
        return world_index_z == 0 || world_index_z + 1 >= SimulationDomain::dimensions.z;
    }
    void SyncVelocitiesZ();
    void SyncVelocities();
    void SyncGridValues(std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* grid,
                        SyncScope scope = SyncScope::Full);

    void SetInitialTemperature();
    void PCG();
    void ProjectVelocities();

    void DrawGrid(Core::RenderPipeline* renderer, bool persistent) const;
    void DrawBoundaryVelocities(Core::RenderPipeline* renderer, bool persistent) const;
    void DrawInterpolatedVelocities(Core::RenderPipeline* renderer, bool persistent) const;

    math::vec3<float> ToWorldCoordinates(size_t face_index, const math::vec3<float>& pos) const;

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
    int cell_sample_count = 1;
    int max_cell_sample_count = 4;
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
    CudaDispatchBuffers cuda_buffers_;
    CudaGraphs cuda_graphs_;
    utils::DefaultThreadPool* thread_pool_;
    // Calculation related data
    // per cell values are allocated on the heap
    std::unique_ptr<std::array<SimulationDomain, domains>> domains_;

    // Actual values
    double crust_temperature_ = 0;      // K
    double ambient_temperature_ = 2500; // K
    double core_temperature_ = 5000;    // K
    double gravitational_acceleration_ = 9.81;

    // Draw
    std::array<Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df>, 6> render_textures_;
    bool drawGrid = true;
    Core::Engine::Assets::SColor grid_color_{180, 180, 180, 80};

    Core::Engine::Assets::SColor boundary_velocity_color_x_ = Core::Engine::Assets::SColors::Red;
    Core::Engine::Assets::SColor boundary_velocity_color_y_ = Core::Engine::Assets::SColors::Green;
    Core::Engine::Assets::SColor boundary_velocity_color_z_ = Core::Engine::Assets::SColors::Blue;
    Core::Engine::Assets::SColor interpolated_velocity_color_ = Core::Engine::Assets::SColors::Cyan;
};

static_assert(SimulationDomain::dimensions[0] == SimulationDomain::dimensions[1] &&
              "dimensions.x and .y always has to be the same");
static_assert(SimulationDomain::halo_size[0] == SimulationDomain::halo_size[1] &&
              "Due to x/y interweaving halos have to match in that direction");

template <size_t AxisIndex>
constexpr size_t WorldIndex(size_t domain_index, size_t cell_index) {
    return domain_index * (SimulationDomain::dimensions[AxisIndex] - 2 * SimulationDomain::halo_size[AxisIndex]) +
           cell_index;
}
