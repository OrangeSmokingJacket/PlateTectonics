#include "euler_fluid_sim.hpp"

#include <Core/Renderer/renderer.hpp>
#include <Math/scalar_ops.hpp>

#include <algorithm>

template <Dimensions GridDimensions>
constexpr Index3D ToIndex3D(size_t linear_index) {
    return Index3D{
        linear_index / (GridDimensions.y * GridDimensions.z),
        (linear_index / GridDimensions.z) % GridDimensions.y,
        linear_index % GridDimensions.z
    };
}

template <Dimensions GridDimensions>
constexpr size_t FromIndex3D(Index3D index_3d) {
    return index_3d.x * GridDimensions.y * GridDimensions.z + index_3d.y * GridDimensions.z + index_3d.z;
}

template <Dimensions GridDimensions>
constexpr size_t FromIndex3D(size_t x, size_t y, size_t z) {
    return x * GridDimensions.y * GridDimensions.z + y * GridDimensions.z + z;
}

// z is packed closest
template <size_t... Indices>
std::array<SimulationDomain, sizeof...(Indices)> InitializeDomainArray(std::index_sequence<Indices...>) {
    return {SimulationDomain(ToIndex3D<EulerFluidSim::domains>(Indices))...};
}

#define ForEachDomain(command)                                                                                         \
    for(auto& domain : *domains_) {                                                                                    \
        domain.command;                                                                                                \
    }

EulerFluidSim::EulerFluidSim(Core::RenderPipeline* renderer)
	: domains_(std::make_unique<std::array<SimulationDomain, Cardinality(domains)>>(InitializeDomainArray(std::make_index_sequence<Cardinality(domains)>())))
    , render_texture_(Core::Engine::Assets::AssetManager::Get()->RegisterAsset(std::make_unique<Core::Engine::Assets::Texture3Df>(full_dimensions.x, full_dimensions.y, full_dimensions.z))) {
	world_origin_offset_.x = -(full_dimensions.x * SimulationConstants::scale * 0.5);
	world_origin_offset_.y = -(full_dimensions.y * SimulationConstants::scale * 0.5);
	world_origin_offset_.z = -(full_dimensions.z * SimulationConstants::scale * 0.5);

	DebugRandomizeBorderVelocities();
    SetInitialTemperature();

    SyncCoefficients();
    SyncGridValues(&SimulationDomain::preconditioner_);

    if (drawGrid) {
        DrawGrid(renderer, true);
    }
}

void EulerFluidSim::Update(float delta_time) {
	// Advection
    ForEachDomain(AdvectVelocities());
    SyncVelocities();
    ForEachDomain(AdvectTemperature());
    SyncGridValues(&SimulationDomain::cell_temperature_);
	// Forces
    ForEachDomain(ApplyBuoyancy(this));
    SyncVelocities();
	// Projection
	ProjectVelocities();

    SetupRenderTexture();
}

void EulerFluidSim::RenderDebug(Core::RenderPipeline* renderer) const {
    // Draw calls
    if (draw_boundary_velocities) {
        DrawBoundaryVelocities(renderer, false);
    }
    if (draw_interpolated_velocities) {
        DrawInterpolatedVelocities(renderer, false);
    }
}

void EulerFluidSim::SetupRenderTexture() {
    if (draw_slice == DrawSlice::None) {
        return;
    }
    // combining multiple slices results in a visual mess, so only one is rendered
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();

    std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* grid = nullptr;
    Core::Engine::Assets::Color* min_color = nullptr;
    Core::Engine::Assets::Color* max_color = nullptr;

    switch (draw_slice) {
    case DrawSlice::Temperature:
        grid = &SimulationDomain::cell_temperature_;
        min_color = &min_temperature_color;
        max_color = &max_temperature_color;
        break;
    case DrawSlice::Pressure:
        grid = &SimulationDomain::cell_pressure_;
        min_color = &min_pressure_color;
        max_color = &max_pressure_color;
        break;
    case DrawSlice::PressureDivergence:
        grid = &SimulationDomain::cell_pressure_divergence_;
        min_color = &min_divergence_color;
        max_color = &max_divergence_color;
        break;
    default:
        assert(false && "draw_slice variant is not handled om SetupRenderTexure");
    }

    // find min/max
    for (const auto& domain : *domains_) {
        for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
            for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                    min = std::min(min, (*(domain.*grid))[x][y][z]);
                    max = std::max(max, (*(domain.*grid))[x][y][z]);
                }
            }
        }
    }

    for (const auto& domain : *domains_) {
        for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
            for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                    float t = ((*(domain.*grid))[x][y][z] - min) / (max - min);
                    auto world_index = WorldIndex(domain.domain_index_, Index3D{x, y, z});
                    // TODO: remove const cast when mutable assets will be implemented
                    const_cast<Core::Engine::Assets::Texture3Df*>(render_texture_.GetAsset())->At(world_index.x, world_index.y, world_index.z) = math::lerp(*min_color, *max_color, t);
                }
            }
        }
    }
}

const Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df>& EulerFluidSim::RenderTexture() const {
    return render_texture_;
}

void EulerFluidSim::DebugRandomizeBorderVelocities() {
	// TODO: Use smooth noise function to get more realistic results
	// Could be optimized, but it is a debug function anyway
    for (auto& domain : *domains_) {
        for (auto it = domain.velocities_x_->flat_begin(); it != domain.velocities_x_->flat_end(); ++it) {
            *it = (math::random<double>() * 2 - 1.0) * SimulationConstants::scale;
        }
        for (auto it = domain.velocities_y_->flat_begin(); it != domain.velocities_y_->flat_end(); ++it) {
            *it = (math::random<double>() * 2 - 1.0) * SimulationConstants::scale;
        }
        for (auto it = domain.velocities_z_->flat_begin(); it != domain.velocities_z_->flat_end(); ++it) {
            *it = (math::random<double>() * 2 - 1.0) * SimulationConstants::scale;
        }

        // Set all solid/solid and solid/fluid borders to 0
        for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
            for (size_t y = 0; y < SimulationDomain::dimensions.y; ++y) {
                for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                    if (IsSolidCell(WorldIndex(domain.domain_index_, Index3D{x, y, z}))) {
                        (*domain.velocities_x_)[x][y][z] = 0.0;
                        (*domain.velocities_x_)[x + 1][y][z] = 0.0;
                        (*domain.velocities_y_)[x][y][z] = 0.0;
                        (*domain.velocities_y_)[x][y + 1][z] = 0.0;
                        (*domain.velocities_z_)[x][y][z] = 0.0;
                        (*domain.velocities_z_)[x][y][z + 1] = 0.0;
                    }
                }
            }
        }
    }

    SyncVelocities();
    for (auto& domain : *domains_) {
        domain.velocities_x_temp_->CopyFrom(*domain.velocities_x_);
        domain.velocities_y_temp_->CopyFrom(*domain.velocities_y_);
        domain.velocities_z_temp_->CopyFrom(*domain.velocities_z_);
    }
}


static constexpr std::array directions{
    math::vec3<int>{-1, -1, -1},
    math::vec3<int>{-1, -1, 0},
    math::vec3<int>{-1, -1, 1},
    math::vec3<int>{-1, 0, -1},
    math::vec3<int>{-1, 0, 0},
    math::vec3<int>{-1, 0, 1},
    math::vec3<int>{-1, 1, -1},
    math::vec3<int>{-1, 1, 0},
    math::vec3<int>{-1, 1, 1},
    math::vec3<int>{0, -1, -1},
    math::vec3<int>{0, -1, 0},
    math::vec3<int>{0, -1, 1},
    math::vec3<int>{0, 0, -1},
    //math::vec3<int>{0, 0, 0}, skip itself
    math::vec3<int>{0, 0, 1},
    math::vec3<int>{0, 1, -1},
    math::vec3<int>{0, 1, 0},
    math::vec3<int>{0, 1, 1},
    math::vec3<int>{1, -1, -1},
    math::vec3<int>{1, -1, 0},
    math::vec3<int>{1, -1, 1},
    math::vec3<int>{1, 0, -1},
    math::vec3<int>{1, 0, 0},
    math::vec3<int>{1, 0, 1},
    math::vec3<int>{1, 1, -1},
    math::vec3<int>{1, 1, 0},
    math::vec3<int>{1, 1, 1},
};

template <math::vec3<size_t> Stagger>
static constexpr std::pair<size_t, size_t> DestinationRange(int dir, size_t axis_index) {
    switch (dir) {
    case -1:
        return {0, SimulationDomain::halo_size[axis_index]};
    case 1:
        return {SimulationDomain::dimensions[axis_index] - SimulationDomain::halo_size[axis_index], SimulationDomain::halo_size[axis_index] + Stagger[axis_index]};
    case 0:
        return {SimulationDomain::halo_size[axis_index], SimulationDomain::stripped_dimensions[axis_index] + Stagger[axis_index]};
    }
    throw std::runtime_error("DestinationRange got incorrect direction");
}

template <math::vec3<size_t> Stagger>
static constexpr  size_t SourceAdjustment(int dir, size_t axis_index) {
    switch (dir) {
    case -1:
        return SimulationDomain::stripped_dimensions[axis_index];
    case 1:
        return SimulationDomain::halo_size[axis_index];
    case 0:
        return SimulationDomain::halo_size[axis_index];
    }
    throw std::runtime_error("SourceAdjustment got incorrect direction");
}

// for MAC grids correct Stagger has to be passed
template <typename Grid, math::vec3<size_t> Stagger = math::vec3<size_t>{0, 0, 0}>
void SyncGridValues(std::array<SimulationDomain, Cardinality(EulerFluidSim::domains)>& domains, std::unique_ptr<Grid> SimulationDomain::* grid) {
    for (size_t dx = 0; dx < EulerFluidSim::domains.x; ++dx) {
        for (size_t dy = 0; dy < EulerFluidSim::domains.y; ++dy) {
            for (size_t dz = 0; dz < EulerFluidSim::domains.z; ++dz) {
                for (const auto& dir : directions) {
                    if (static_cast<int>(dx) + dir.x < static_cast<int>(0) ||
                        static_cast<int>(dy) + dir.y < static_cast<int>(0) ||
                        static_cast<int>(dz) + dir.z < static_cast<int>(0) ||
                        static_cast<int>(dx) + dir.x >= static_cast<int>(EulerFluidSim::domains.x) ||
                        static_cast<int>(dy) + dir.y >= static_cast<int>(EulerFluidSim::domains.y) ||
                        static_cast<int>(dz) + dir.z >= static_cast<int>(EulerFluidSim::domains.z)) {
                            continue;
                        }

                    auto [hx, cx] = DestinationRange<Stagger>(dir.x, 0);
                    auto [hy, cy] = DestinationRange<Stagger>(dir.y, 1);
                    auto [hz, cz] = DestinationRange<Stagger>(dir.z, 2);

                    auto sx = SourceAdjustment<Stagger>(dir.x, 0);
                    auto sy = SourceAdjustment<Stagger>(dir.y, 1);
                    auto sz = SourceAdjustment<Stagger>(dir.z, 2);

                    for (size_t i = 0; i < cx; ++i) {
                        for (size_t j = 0; j < cy; ++j) {
                            for (size_t k = 0; k < cz; ++k) {
                                (*(domains[FromIndex3D<EulerFluidSim::domains>(dx, dy, dz)].*grid))[hx + i][hy + j][hz + k] =
                                    (*(domains[FromIndex3D<EulerFluidSim::domains>(dx + dir.x, dy + dir.y, dz + dir.z)].*grid))[sx + i][sy + j][sz + k];
                            }
                        }
                    }
                }
            }
        }
    }
}

void EulerFluidSim::SyncVelocitiesX() {
    ::SyncGridValues<SimulationDomain::GridXValues, math::vec3<size_t>{1, 0, 0}>(*domains_, &SimulationDomain::velocities_x_);
}

void EulerFluidSim::SyncVelocitiesY() {
    ::SyncGridValues<SimulationDomain::GridYValues, math::vec3<size_t>{0, 1, 0}>(*domains_, &SimulationDomain::velocities_y_);
}

void EulerFluidSim::SyncVelocitiesZ() {
    ::SyncGridValues<SimulationDomain::GridZValues, math::vec3<size_t>{0, 0, 1}>(*domains_, &SimulationDomain::velocities_z_);
}

void EulerFluidSim::SyncVelocities() {
    SyncVelocitiesX();
    SyncVelocitiesY();
    SyncVelocitiesZ();
}

void EulerFluidSim::SyncGridValues(std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* grid) {
    ::SyncGridValues<SimulationDomain::GridValues>(*domains_, grid);
}

void EulerFluidSim::SyncCoefficients() {
    ::SyncGridValues<SimulationDomain::GridCoefficientValues>(*domains_, &SimulationDomain::coefficient_);
}

void EulerFluidSim::SetInitialTemperature() {
    for (auto& domain : *domains_) {
        // set all to ambient, except lower boundary
        domain.cell_temperature_->SetTo(ambient_temperature_);

        if (domain.domain_index_.y == 0) {
            for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
                for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                    (*domain.cell_temperature_)[x][0][z] = core_temperature_;
                }
            }
        } else if (domain.domain_index_.y == domains.y - 1) {
            for (size_t x = 0; x < SimulationDomain::dimensions.x; ++x) {
                for (size_t z = 0; z < SimulationDomain::dimensions.z; ++z) {
                    (*domain.cell_temperature_)[x][SimulationDomain::dimensions.y - 1][z] = crust_temperature_;
                }
            }
        }

        domain.cell_temperature_temp_->CopyFrom(*domain.cell_temperature_);
    }
}


void EulerFluidSim::PCG() {
    ForEachDomain(cell_pressure_->SetToZero());
	// If divergence is low enough zero pressure will be close enough
    // It might be faster in some cases to skip this check
	double maxDivergence = std::numeric_limits<double>::lowest();
	double minDivergence = std::numeric_limits<double>::max();
    for (const auto& domain : *domains_) {
        auto min_max = domain.DivergenceMinMax();
        minDivergence = std::min(minDivergence, min_max.first);
        maxDivergence = std::max(maxDivergence, min_max.second);
    }
	if (fabs(maxDivergence) < std::numeric_limits<double>::epsilon() && fabs(minDivergence) < std::numeric_limits<double>::epsilon())
		return;

    ForEachDomain(residual_->CopyFrom(*domain.cell_pressure_divergence_));
    ForEachDomain(ApplyPreconditioner());
    SyncGridValues(&SimulationDomain::auxiliary_);
    ForEachDomain(search_->CopyFrom(*domain.auxiliary_));

	double sigma = 0;
    for (const auto& domain : *domains_) {
        sigma += domain.RealCellsDotProduct(&SimulationDomain::auxiliary_, &SimulationDomain::residual_);
    }

	for (last_PCG_iteration_count = 0; last_PCG_iteration_count < SimulationConstants::max_solver_iterations; last_PCG_iteration_count++) {
	    assert(!std::isinf(sigma));
	    assert(!std::isnan(sigma));
	    ForEachDomain(ApplyCoefficients());
        SyncGridValues(&SimulationDomain::auxiliary_);

	    double dot_product = 0;
	    for (const auto& domain : *domains_) {
	        dot_product += domain.RealCellsDotProduct(&SimulationDomain::auxiliary_, &SimulationDomain::search_);
	    }
	    double alpha = sigma / dot_product;
	    for (auto& domain : *domains_) {
	        for (size_t j = 0; j < domain.cell_pressure_->cardinality; j++)
	        {
	            *(domain.cell_pressure_->flat_data() + j) += *(domain.search_->flat_data() + j) * alpha;
	            *(domain.residual_->flat_data() + j) -= *(domain.auxiliary_->flat_data() + j) * alpha;
	        }
	    }

	    bool result_is_good_enough = true;
	    for (const auto& domain : *domains_) {
	        if (domain.residual_->Length() > SimulationConstants::solver_tolerance) {
	            result_is_good_enough = false;
	            break;
	        }
	    }

		if (result_is_good_enough)
			break;

	    ForEachDomain(ApplyPreconditioner());
        SyncGridValues(&SimulationDomain::auxiliary_);

	    double sigma_new = 0;
	    for (const auto& domain : *domains_) {
            sigma_new += domain.RealCellsDotProduct(&SimulationDomain::auxiliary_, &SimulationDomain::residual_);
	    }
		double beta = sigma_new / sigma;
	    for (auto& domain : *domains_) {
	        for (size_t j = 0; j < domain.search_->cardinality; j++)
	        {
	            *(domain.search_->flat_data() + j) = *(domain.auxiliary_->flat_data() + j) + *(domain.search_->flat_data() + j) * beta;
	        }
	    }

	    //assert(sigma_new < sigma);
		sigma = sigma_new;
	}
	// TODO: Log used iterations properly
}

void EulerFluidSim::ProjectVelocities() {
    ForEachDomain(CalculateDivergence());
    SyncGridValues(&SimulationDomain::cell_pressure_divergence_);
	PCG();
    ForEachDomain(UpdateVelocities());
    SyncVelocities();
}

// Debug draw methods
void EulerFluidSim::DrawGrid(Core::RenderPipeline* renderer, bool persistent) const {
	if (boundaryOnly) {
		auto box = math::vec3<float>{full_dimensions.x - 2, full_dimensions.y - 2, full_dimensions.z - 2} * SimulationConstants::scale * 0.5f;
		renderer->DebugDrawBox(math::vec3<float>{}, box, grid_color_, persistent);
	} else {
		math::vec3<float> lineStart = world_origin_offset_;
		math::vec3<float> lineEnd{world_origin_offset_.x, world_origin_offset_.y,world_origin_offset_.z + full_dimensions.z * SimulationConstants::scale};
		for (uint64_t x = 0; x <= full_dimensions.x; ++x) {
			lineStart.x = world_origin_offset_.x + x * SimulationConstants::scale;
			lineEnd.x = world_origin_offset_.x + x * SimulationConstants::scale;
			for (uint64_t y = 0; y <= full_dimensions.y; ++y) {
				lineStart.y = world_origin_offset_.y + y * SimulationConstants::scale;
				lineEnd.y = world_origin_offset_.y + y * SimulationConstants::scale;
				renderer->DebugDrawLine(lineStart, lineEnd, grid_color_, persistent);
			}
			lineStart.y = world_origin_offset_.y;
			// lineEnd is already set
			// lineEnd.y = world_origin_offset_.y + dimensions.y * SimulationConstants::scale;
			for (uint64_t z = 0; z <= full_dimensions.z; ++z) {
				lineStart.z = world_origin_offset_.z + z * SimulationConstants::scale;
				lineEnd.z = world_origin_offset_.z + z * SimulationConstants::scale;
				renderer->DebugDrawLine(lineStart, lineEnd, grid_color_, persistent);
			}
			lineStart.z = world_origin_offset_.z;
			// lineEnd is already set
			// lineEnd.z = world_origin_offset_.z + dimensions.z * SimulationConstants::scale;
		}
		lineStart.x = world_origin_offset_.x;
		// lineEnd is already set
		// lineEnd.x = world_origin_offset_.x + dimensions.x * SimulationConstants::scale;
		for (uint64_t y = 0; y <= full_dimensions.y; ++y) {
			lineStart.y = world_origin_offset_.y + y * SimulationConstants::scale;
			lineEnd.y = world_origin_offset_.y + y * SimulationConstants::scale;
			for (uint64_t z = 0; z <= full_dimensions.z; ++z) {
				lineStart.z = world_origin_offset_.z + z * SimulationConstants::scale;
				lineEnd.z = world_origin_offset_.z + z * SimulationConstants::scale;
				renderer->DebugDrawLine(lineStart, lineEnd, grid_color_, persistent);
			}
		}
	}
}

void EulerFluidSim::DrawBoundaryVelocities(Core::RenderPipeline* renderer, bool persistent) const {
    // TODO: skip halo cells
    for (const auto& domain : *domains_) {
	    for (uint64_t x = SimulationDomain::halo_size.x; x < SimulationDomain::dimensions.x - SimulationDomain::halo_size.x; ++x) {
		    for (uint64_t y = SimulationDomain::halo_size.y; y < SimulationDomain::dimensions.y - SimulationDomain::halo_size.y; ++y) {
			    for (uint64_t z = SimulationDomain::halo_size.z; z < SimulationDomain::dimensions.z - SimulationDomain::halo_size.z; ++z) {
			        auto world_index = WorldIndex(domain.domain_index_, Index3D{x, y, z});
			        math::vec3<float> start = world_origin_offset_ + SimulationConstants::scale *
                        (static_cast<math::vec3<float>>(world_index) + math::vec3<float>{0.0f, 0.5f, 0.5f});
				    double arrow_value = (*domain.velocities_x_)[x][y][z];
				    renderer->DebugDrawArrow(start,
										     start + math::vec3<float>{arrow_value * 0.5, 0.0, 0.0},
										      fabsf(arrow_value) * 0.25f,
										     boundary_velocity_color_x_,
										     persistent);
			        start.x += SimulationConstants::scale * 0.5f;
			        start.y -= SimulationConstants::scale * 0.5f;
				    arrow_value = (*domain.velocities_y_)[x][y][z];
				    renderer->DebugDrawArrow(start,
										     start + math::vec3<float>{0.0, arrow_value * 0.5, 0.0},
										     fabsf(arrow_value) * 0.25f,
										     boundary_velocity_color_y_,
										     persistent);
			        start.z -= SimulationConstants::scale * 0.5f;
				    arrow_value = (*domain.velocities_z_)[x][y][z];
				    renderer->DebugDrawArrow(start,
										     start + math::vec3<float>{0.0, 0.0, arrow_value * 0.5},
										     fabsf(arrow_value) * 0.25f,
										     boundary_velocity_color_z_,
										     persistent);
			    }
		    }
	    }
    }
}

void EulerFluidSim::DrawInterpolatedVelocities(Core::RenderPipeline* renderer, bool persistent) const
{
	math::vec3<float> nPos;
	math::vec3<float> interpolationPos;
	// border cells will have 0 velocity anyway, so we can skip them
    // TODO: skip halo cells
	if (cell_subdivision == 1) {
	    for (const auto& domain : *domains_) {
	        for (uint64_t x = 1; x < SimulationDomain::dimensions.x - 1; ++x) {
	            for (uint64_t y = 1; y < SimulationDomain::dimensions.y - 1; ++y) {
	                for (uint64_t z = 1; z < SimulationDomain::dimensions.z - 1; ++z) {
	                    nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} + ValueGridOffset;
	                    interpolationPos = domain.DenormalizePosition(nPos);
	                    math::vec3<float> velocity = domain.SampleVelocity(nPos);
	                    interpolationPos +=  world_origin_offset_ +static_cast<math::vec3<float>>(domain.domain_index_ * SimulationDomain::stripped_dimensions) * SimulationConstants::scale;
	                    renderer->DebugDrawArrow(interpolationPos,
                                                 interpolationPos + velocity,
                                                 velocity.magnitude() * 0.25f,
                                                 interpolated_velocity_color_,
                                                 persistent);
	                }
	            }
	        }
	    }
	} else {
	    for (const auto& domain : *domains_) {
	        for (uint64_t x = cell_subdivision; x <= (SimulationDomain::dimensions.x - 1) * cell_subdivision; ++x) {
	            for (uint64_t y = cell_subdivision; y <= (SimulationDomain::dimensions.y - 1) * cell_subdivision; ++y) {
	                for (uint64_t z = cell_subdivision; z <= (SimulationDomain::dimensions.z - 1) * cell_subdivision; ++z) {
	                    nPos = math::vec3<float>{
	                        static_cast<float>(x) / static_cast<float>(cell_subdivision) + 1.0f / static_cast<float>(cell_subdivision + 1),
                            static_cast<float>(y) / static_cast<float>(cell_subdivision) + 1.0f / static_cast<float>(cell_subdivision + 1),
                            static_cast<float>(z) / static_cast<float>(cell_subdivision) + 1.0f / static_cast<float>(cell_subdivision + 1)};
	                    interpolationPos = domain.DenormalizePosition(nPos);
	                    math::vec3<float> velocity = domain.SampleVelocity(nPos) / static_cast<float>(cell_subdivision);
	                    interpolationPos +=  world_origin_offset_ +static_cast<math::vec3<float>>(domain.domain_index_ * SimulationDomain::stripped_dimensions) * SimulationConstants::scale;
	                    renderer->DebugDrawArrow(interpolationPos,
                                                 interpolationPos + velocity,
                                                 velocity.magnitude() * 0.25f,
                                                 interpolated_velocity_color_,
                                                 persistent);
	                }
	            }
	        }
	    }
	}
}
