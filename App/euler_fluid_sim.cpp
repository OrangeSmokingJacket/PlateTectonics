#include "euler_fluid_sim.hpp"

#include <Math/scalar_ops.hpp>

#include <algorithm>

#include "Core/Renderer/renderer.hpp"

static const math::vec3<float> ValueGridOffset{0.5, 0.5, 0.5};
static const math::vec3<float> VelocityXGridOffset{0.0, 0.5, 0.5};
static const math::vec3<float> VelocityYGridOffset{0.5, 0.0, 0.5};
static const math::vec3<float> VelocityZGridOffset{0.5, 0.5, 0.0};

// Solver is based upon FLUID SIMULATION SIGGRAPH 2007 Course Notes

// Sets default values
// TODO still assumes that z is up from UE
EulerFluidSim::EulerFluidSim(Core::RenderPipeline* renderer)
	: cell_pressure_(std::make_unique<GridValues>())
	, cell_temperature_(std::make_unique<GridValues>())
	, cell_temperature_temp_(std::make_unique<GridValues>())
	, velocities_x_(std::make_unique<GridXValues>())
	, velocities_x_temp_(std::make_unique<GridXValues>())
	, velocities_y_(std::make_unique<GridYValues>())
	, velocities_y_temp_(std::make_unique<GridYValues>())
	, velocities_z_(std::make_unique<GridZValues>())
	, velocities_z_temp_(std::make_unique<GridZValues>())
	, coefficient_(std::make_unique<GridCoefficientValues>())
	, preconditioner_(std::make_unique<GridValues>())
	, intermediate_(std::make_unique<GridValues>())
	, cell_pressure_divergence_(std::make_unique<GridValues>())
	, residual_(std::make_unique<GridValues>())
	, auxiliary_(std::make_unique<GridValues>())
	, search_(std::make_unique<GridValues>())
    , render_texture_(Core::Engine::Assets::AssetManager::Get()->RegisterAsset(std::make_unique<Core::Engine::Assets::Texture3Df>(dimensions.x, dimensions.y, dimensions.z)))
{	
	world_origin_offset_.x = -(dimensions.x * scale * 0.5);
	world_origin_offset_.y = -(dimensions.y * scale * 0.5);
	world_origin_offset_.z = -(dimensions.z * scale * 0.5);
	
	cell_pressure_->SetToZero();
	coefficient_->SetToZero();
	preconditioner_->SetToZero();
	intermediate_->SetToZero();
	cell_pressure_divergence_->SetToZero();
	residual_->SetToZero();
	auxiliary_->SetToZero();
	search_->SetToZero();
	
	// Since geometry is static, coefficients and preconditioner could be calculated only once
	CalculateCoefficients();
	CalculatePreconditioner();
	
	//velocities_x_->SetToZero();
	//velocities_y_->SetToZero();
	//velocities_z_->SetToZero();
	//velocities_x_temp_->SetToZero();
	//velocities_y_temp_->SetToZero();
	//velocities_z_temp_->SetToZero();
	DebugRandomizeBorderVelocities();
	
    SetInitialTemperature();

    if (drawGrid) {
        DrawGrid(renderer, true);
    }
}

void EulerFluidSim::Update(float delta_time)
{
	// Advection
	AdvectVelocities();
	AdvectTemperature();
	// Forces
	ApplyBuoyancy();
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

    GridValues* grid = nullptr;
    Core::Engine::Assets::Color* min_color = nullptr;
    Core::Engine::Assets::Color* max_color = nullptr;

    switch (draw_slice) {
    case DrawSlice::Temperature:
        grid = cell_temperature_.get();
        min_color = &min_temperature_color;
        max_color = &max_temperature_color;
        break;
    case DrawSlice::Pressure:
        grid = cell_pressure_.get();
        min_color = &min_pressure_color;
        max_color = &max_pressure_color;
        break;
    case DrawSlice::PressureDivergence:
        grid = cell_pressure_divergence_.get();
        min_color = &min_divergence_color;
        max_color = &max_divergence_color;
        break;
    default:
        assert(false && "draw_slice variant is not handled om SetupRenderTexure");
    }

    // find min/max
    for (size_t x = 0; x < dimensions.x; ++x) {
        for (size_t y = 0; y < dimensions.y; ++y) {
            for (size_t z = 0; z < dimensions.z; ++z) {
                min = std::min(min, (*grid)[x][y][z]);
                max = std::max(max, (*grid)[x][y][z]);
            }
        }
    }

    for (size_t x = 0; x < dimensions.x; ++x) {
        for (size_t y = 0; y < dimensions.y; ++y) {
            for (size_t z = 0; z < dimensions.z; ++z) {
                float t = ((*grid)[x][y][z] - min) / (max - min);
                // TODO: remove const cast when mutable assets will be implemented
                const_cast<Core::Engine::Assets::Texture3Df*>(render_texture_.GetAsset())->At(x, y, z) = math::lerp(*min_color, *max_color, t);
            }
        }
    }
    }

const Core::Engine::Assets::AssetRef<Core::Engine::Assets::Texture3Df>& EulerFluidSim::RenderTexture() const {
    return render_texture_;
}

void EulerFluidSim::DebugRandomizeBorderVelocities()
{
	// TODO: Use smooth noise function to get more realistic results
	// Could be optimized, but it is a debug function anyway
	for (auto it = velocities_x_->flat_begin(); it != velocities_x_->flat_end(); ++it)
	{
		*it = (math::random<double>() * 2 - 1.0) * scale;
	}
	for (auto it = velocities_y_->flat_begin(); it != velocities_y_->flat_end(); ++it)
	{
		*it = (math::random<double>() * 2 - 1.0) * scale;
	}
	for (auto it = velocities_z_->flat_begin(); it != velocities_z_->flat_end(); ++it)
	{
		*it = (math::random<double>() * 2 - 1.0) * scale;
	}
	
	// Set all solid/solid and solid/fluid borders to 0
	for (uint64_t x = 0; x < dimensions.x; ++x)
	{
		for (uint64_t y = 0; y < dimensions.y; ++y)
		{
			for (uint64_t z = 0; z < dimensions.z; ++z)
			{
				if (IsSolidCell(x, y, z))
				{
					(*velocities_x_)[x][y][z] = 0.0;
					(*velocities_x_)[x + 1][y][z] = 0.0;
					(*velocities_y_)[x][y][z] = 0.0;
					(*velocities_y_)[x][y + 1][z] = 0.0;
					(*velocities_z_)[x][y][z] = 0.0;
					(*velocities_z_)[x][y][z + 1] = 0.0;
				}
			}
		}
	}
	
	velocities_x_temp_->CopyFrom(*velocities_x_);
	velocities_y_temp_->CopyFrom(*velocities_y_);
	velocities_z_temp_->CopyFrom(*velocities_z_);
}

template <typename Grid>
Grid::value_type TrilinearInterpolation(const Grid* grid, const math::vec3<float>& normalizedPos)
{
	int indexX = floor(normalizedPos.x);
	int indexY = floor(normalizedPos.y);
	int indexZ = floor(normalizedPos.z);
	
	double fractionX = normalizedPos.x - indexX;
	double fractionY = normalizedPos.y - indexY;
	double fractionZ = normalizedPos.z - indexZ;
	
	// Actual interpolation
	typename Grid::value_type c00 = math::lerp((*grid)[indexX][indexY][indexZ], (*grid)[indexX + 1][indexY][indexZ], fractionX);
	typename Grid::value_type c01 = math::lerp((*grid)[indexX][indexY][indexZ + 1], (*grid)[indexX + 1][indexY][indexZ + 1], fractionX);
	typename Grid::value_type c10 = math::lerp((*grid)[indexX][indexY + 1][indexZ], (*grid)[indexX + 1][indexY + 1][indexZ], fractionX);
	typename Grid::value_type c11 = math::lerp((*grid)[indexX][indexY + 1][indexZ + 1], (*grid)[indexX + 1][indexY + 1][indexZ + 1], fractionX);
	
	typename Grid::value_type c0 = math::lerp(c00, c10, fractionY);
	typename Grid::value_type c1 = math::lerp(c01, c11, fractionY);
	
	return math::lerp(c0, c1, fractionZ);
}

template <typename Value>
Value CubicInterpolation(const Value& q0, const Value& q1, const Value& q2, const Value& q3, double x)
{
	// TODO: option to disable safety measures (less reliable in advection, but more accurate)
	Value d0 = (q2  - q0) * 0.5;
	Value d1 = (q3  - q1) * 0.5;
	
	Value delta_q = q2 - q1;
	// safety measure to guarantee interpolation stability
	if (math::is_equal(delta_q, 0.0) || std::signbit(delta_q) != std::signbit(d0) || std::signbit(delta_q) != std::signbit(d1))
	{
		d0 = 0;
		d1 = 0;
	}
	
	Value result = q1 +
				   d0 * x + 
				   (3 * delta_q - 2 * d0 - d1) * math::square(x) +
				   (-2 * delta_q + d0 + d1) * math::cube(x);
	return result;
}

template <typename Grid>
Grid::value_type CubicInterpolation(const Grid* grid, const math::vec3<float>& normalizedPos)
{
	// 2 cells border will make this check redundant
	auto gridValue = [grid](int x, int y, int z) -> typename Grid::value_type
	{
		if (x < 0 || y < 0 || z < 0 ||
		    x >= static_cast<int>(Grid::size_X) ||
		    y >= static_cast<int>(Grid::size_Y) ||
		    z >= static_cast<int>(Grid::size_Z))
			return {};
		return (*grid)[x][y][z];
	};

	int indexX = floor(normalizedPos.x);
	int indexY = floor(normalizedPos.y);
	int indexZ = floor(normalizedPos.z);

	double fractionX = normalizedPos.x - indexX;
	double fractionY = normalizedPos.y - indexY;
	double fractionZ = normalizedPos.z - indexZ;

	typename Grid::value_type xValues[4][4];
	typename Grid::value_type yValues[4];

	for (size_t i = 0; i < 4; i++)
	{
		for (size_t j = 0; j < 4; j++)
		{
			const int _y = static_cast<int>(i) - 1;
			const int _z = static_cast<int>(j) - 1;
			xValues[i][j] = CubicInterpolation(gridValue(indexX - 1, indexY + _y, indexZ + _z),
								 			   gridValue(indexX, indexY + _y, indexZ + _z),
								 			   gridValue(indexX + 1, indexY + _y, indexZ + _z),
								 			   gridValue(indexX + 2, indexY + _y, indexZ + _z),
								 			   fractionX);
		}
	}

	for (size_t i = 0; i < 4; i++)
	{
		yValues[i] = CubicInterpolation(xValues[0][i], xValues[1][i], xValues[2][i], xValues[3][i], fractionY);
	}

	return CubicInterpolation(yValues[0], yValues[1], yValues[2], yValues[3], fractionZ);
}

// TODO: make switching constexpr
template <typename Grid>
Grid::value_type Interpolation(const Grid* grid, const math::vec3<float>& normalizedPos, EulerFluidSim::InterpolationType type)
{
	switch (type) {
		case EulerFluidSim::InterpolationType::Cubic:
			return CubicInterpolation(grid, normalizedPos);
		case EulerFluidSim::InterpolationType::Trilinear:
		default:
			return TrilinearInterpolation(grid, normalizedPos);
	}
}

math::vec3<float> EulerFluidSim::NormalizePosition(const math::vec3<float>& pos) const
{
	return (pos - world_origin_offset_) / scale;
}

math::vec3<float> EulerFluidSim::DenormalizePosition(const math::vec3<float>& nPos) const
{
	return nPos * scale + world_origin_offset_;
}

math::vec3<float> EulerFluidSim::SampleVelocity(const math::vec3<float>& nPos) const
{
	math::vec3<float> result;
	result.x = Interpolation(velocities_x_.get(), nPos - VelocityXGridOffset, interpolation_type);
	result.y = Interpolation(velocities_y_.get(), nPos - VelocityYGridOffset, interpolation_type);
	result.z = Interpolation(velocities_z_.get(), nPos - VelocityZGridOffset, interpolation_type);
	return result;
}

constexpr bool EulerFluidSim::IsSolidCell(size_t x, size_t y, size_t z)
{
	return x == 0 || y == 0 || z == 0 || x + 1 >= dimensions.x || y + 1 >= dimensions.y || z + 1 >= dimensions.z;
}

void EulerFluidSim::SetInitialTemperature()
{
	// set all to ambient, except lower boundary
	cell_temperature_->SetTo(ambient_temperature_);
	for (size_t x = 0; x < dimensions.x; ++x)
	{
		for (size_t z = 0; z < dimensions.z; ++z)
		{
			(*cell_temperature_)[x][0][z] = core_temperature_;
			(*cell_temperature_)[x][dimensions.y - 1][z] = 0;
		}
	}
	cell_temperature_temp_->CopyFrom(*cell_temperature_);
}

// Some useful macros to maintain readability
#define Adiag(x, y, z) (*coefficient_)[x][y][z].Adiag
#define Aplusx(x, y, z) (*coefficient_)[x][y][z].Aplusx
#define Aplusy(x, y, z) (*coefficient_)[x][y][z].Aplusy
#define Aplusz(x, y, z) (*coefficient_)[x][y][z].Aplusz
#define prec(x, y, z) (*preconditioner_)[x][y][z]
#define intermediate(x, y, z) (*intermediate_)[x][y][z]
#define aux(x, y, z) (*auxiliary_)[x][y][z]
#define search(x, y, z) (*search_)[x][y][z]

void EulerFluidSim::CalculateCoefficients()
{
	static constexpr double k = fixed_time_step / (density * scale * scale);

	for (size_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (size_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (size_t z = 1; z < dimensions.z - 1; ++z)
			{
				int flowPosX = !IsSolidCell(x + 1, y, z);
				int flowNegX = !IsSolidCell(x - 1, y, z);
				int flowPosY = !IsSolidCell(x, y + 1, z);
				int flowNegY = !IsSolidCell(x, y - 1, z);
				int flowPosZ = !IsSolidCell(x, y, z + 1);
				int flowNegZ = !IsSolidCell(x, y, z - 1);

				int flowSum = flowPosX + flowNegX + flowPosY + flowNegY + flowPosZ + flowNegZ;

				// Pressure coefficients
				Adiag(x, y, z) = flowSum * k;
				Aplusx(x, y, z) = -flowPosX * k;
				Aplusy(x, y, z) = -flowPosY * k;
				Aplusz(x, y, z) = -flowPosZ * k;
				Aplusx(x - 1, y, z) = -flowNegX * k;
				Aplusy(x, y - 1, z) = -flowNegY * k;
				Aplusz(x, y, z - 1) = -flowNegZ * k;
			}
		}
	}
}

void EulerFluidSim::CalculatePreconditioner()
{
	static constexpr double tuning = 0.97;
	static constexpr double epsilon = 1.0e-30;

	for (size_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (size_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (size_t z = 1; z < dimensions.z - 1; ++z)
			{
				double ax = math::square(Aplusx(x - 1, y, z) * prec(x - 1, y, z));
				double ay = math::square(Aplusy(x, y - 1, z) * prec(x, y - 1, z));
				double az = math::square(Aplusz(x, y, z - 1) * prec(x, y, z - 1));

				double bx = Aplusx(x - 1, y, z) * (Aplusy(x - 1, y, z) + Aplusz(x - 1, y, z)) * math::square(prec(x - 1, y, z));
				double by = Aplusy(x, y - 1, z) * (Aplusx(x, y - 1, z) + Aplusz(x, y - 1, z)) * math::square(prec(x, y - 1, z));
				double bz = Aplusz(x, y, z - 1) * (Aplusx(x, y, z - 1) + Aplusy(x, y, z - 1)) * math::square(prec(x, y, z - 1));
				
				double e = Adiag(x, y, z) - ax - ay - az - tuning * (bx + by + bz);
				prec(x, y, z) = 1 / std::sqrt(e + epsilon);
			}
		}
	}
}

void EulerFluidSim::CalculateDivergence()
{
	for (size_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (size_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (size_t z = 1; z < dimensions.z - 1; ++z)
			{
				int flowPosX = !IsSolidCell(x + 1, y, z);
				int flowNegX = !IsSolidCell(x - 1, y, z);
				int flowPosY = !IsSolidCell(x, y + 1, z);
				int flowNegY = !IsSolidCell(x, y - 1, z);
				int flowPosZ = !IsSolidCell(x, y, z + 1);
				int flowNegZ = !IsSolidCell(x, y, z - 1);
				
				double velocityX_POS = (*velocities_x_)[x + 1][y][z] * flowPosX;
				double velocityX_NEG = (*velocities_x_)[x][y][z] * flowNegX;
				double velocityY_POS = (*velocities_y_)[x][y + 1][z] * flowPosY;
				double velocityY_NEG = (*velocities_y_)[x][y][z] * flowNegY;
				double velocityZ_POS = (*velocities_z_)[x][y][z + 1] * flowPosZ;
				double velocityZ_NEG = (*velocities_z_)[x][y][z] * flowNegZ;
	
				double gradientX = velocityX_POS - velocityX_NEG;
				double gradientY = velocityY_POS - velocityY_NEG;
				double gradientZ = velocityZ_POS - velocityZ_NEG;
	
				(*cell_pressure_divergence_)[x][y][z] = -(gradientX + gradientY + gradientZ) / scale;
			}
		}
	}
}

void EulerFluidSim::ApplyCoefficients(GridValues& result, const GridValues& apply_to)
{
	for (size_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (size_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (size_t z = 1; z < dimensions.z - 1; ++z)
			{
				result[x][y][z] = Adiag(x, y, z) * apply_to[x][y][z] +
							      Aplusx(x - 1, y, z) * apply_to[x - 1][y][z] +
							      Aplusx(x, y, z) * apply_to[x + 1][y][z] +
							      Aplusy(x, y - 1, z) * apply_to[x][y - 1][z] +
							      Aplusy(x, y, z) * apply_to[x][y + 1][z] +
							      Aplusz(x, y, z - 1) * apply_to[x][y][z - 1] +
							      Aplusz(x, y, z) * apply_to[x][y][z + 1];
			}
		}
	}
}

void EulerFluidSim::ApplyPreconditioner(GridValues& result, const GridValues& apply_to)
{
	for (size_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (size_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (size_t z = 1; z < dimensions.z - 1; ++z)
			{
				double ax = Aplusx(x - 1, y, z) * prec(x - 1, y, z) * intermediate(x - 1, y, z);
				double ay = Aplusy(x, y - 1, z) * prec(x, y - 1, z) * intermediate(x, y - 1, z);
				double az = Aplusz(x, y, z - 1) * prec(x, y, z - 1) * intermediate(x, y, z - 1);
				
				intermediate(x, y, z) = (apply_to[x][y][z] - ax - ay - az) * prec(x, y, z);
			}
		}
	}
	for (size_t x = dimensions.x - 2; x > 0; x--)
	{
		for (size_t y = dimensions.y - 2; y > 0; y--)
		{
			for (size_t z = dimensions.z - 2; z > 0; z--)
			{
				double ax = Aplusx(x, y, z) * prec(x, y, z) * result[x + 1][y][z];
				double ay = Aplusy(x, y, z) * prec(x, y, z) * result[x][y + 1][z];
				double az = Aplusz(x, y, z) * prec(x, y, z) * result[x][y][z + 1];
				
				result[x][y][z] = (intermediate(x, y, z) - ax - ay - az) * prec(x, y, z);
			}
		}
	}
}

void EulerFluidSim::PCG()
{
	cell_pressure_->SetToZero();
	// If divergence is low enough zero pressure will be close enough
	double maxDivergence = std::numeric_limits<double>::lowest();
	double minDivergence = std::numeric_limits<double>::max();
	for (size_t i = 0; i < cell_pressure_divergence_->cardinality; i++)
	{
		maxDivergence = std::max(maxDivergence, *(cell_pressure_divergence_->flat_begin() + i));
		minDivergence = std::min(minDivergence, *(cell_pressure_divergence_->flat_begin() + i));
	}
	if (fabs(maxDivergence) < std::numeric_limits<double>::epsilon() && fabs(minDivergence) < std::numeric_limits<double>::epsilon())
		return;
	
	residual_->CopyFrom(*cell_pressure_divergence_);
	ApplyPreconditioner(*auxiliary_, *residual_);
	search_->CopyFrom(*auxiliary_);
	double sigma = auxiliary_->DotProduct(*residual_);
	for (last_PCG_iteration_count = 0; last_PCG_iteration_count < max_solver_iterations; last_PCG_iteration_count++)
	{
		ApplyCoefficients(*auxiliary_, *search_);
		double alpha = sigma / auxiliary_->DotProduct(*search_);
		for (size_t j = 0; j < cell_pressure_->cardinality; j++)
		{
			*(cell_pressure_->flat_data() + j) += *(search_->flat_data() + j) * alpha;
			*(residual_->flat_data() + j) -= *(auxiliary_->flat_data() + j) * alpha;
		}
		
		if (residual_->Length() < solver_tolerance)
			break;
		
		ApplyPreconditioner(*auxiliary_, *residual_);
		double sigma_new = auxiliary_->DotProduct(*residual_);
		double beta = sigma_new / sigma;
		for (size_t j = 0; j < search_->cardinality; j++)
		{
			*(search_->flat_data() + j) = *(auxiliary_->flat_data() + j) + *(search_->flat_data() + j) * beta;
		}
		sigma = sigma_new;
	}
	// TODO: Log used iterations properly
}

void EulerFluidSim::ProjectVelocities()
{
	CalculateDivergence();
	PCG();
	UpdateVelocities();
}

void EulerFluidSim::UpdateVelocities()
{
	const double k = fixed_time_step / (density * scale);
	
	for (uint64_t x = 0; x < dimensions.x; ++x)
	{
		for (uint64_t y = 0; y < dimensions.y; ++y)
		{
			for (uint64_t z = 0; z < dimensions.z; ++z)
			{
				if (IsSolidCell(x, y, z))
				{
					(*velocities_x_)[x][y][z] = 0;
					(*velocities_x_)[x + 1][y][z] = 0;
					(*velocities_y_)[x][y][z] = 0;
					(*velocities_y_)[x][y + 1][z] = 0;
					(*velocities_z_)[x][y][z] = 0;
					(*velocities_z_)[x][y][z + 1] = 0;
					continue;
				}
				if (!IsSolidCell(x + 1, y, z)) {
					(*velocities_x_)[x + 1][y][z] -= k * ((*cell_pressure_)[x + 1][y][z] - (*cell_pressure_)[x][y][z]);
				}
				if (!IsSolidCell(x, y + 1, z))
				{
					(*velocities_y_)[x][y + 1][z] -= k * ((*cell_pressure_)[x][y + 1][z] - (*cell_pressure_)[x][y][z]);
				}
				if (!IsSolidCell(x, y, z + 1))
				{
					(*velocities_z_)[x][y][z + 1] -= k * ((*cell_pressure_)[x][y][z + 1] - (*cell_pressure_)[x][y][z]);
				}
			}
		}
	}
}

void EulerFluidSim::AdvectVelocities()
{
	// TODO: unified loop?
	// TODO: SampleVelocity actually needed only for 2 axes at a time, since one is always at grid local point
	math::vec3<float> pos;
	math::vec3<float> prevPos;
	math::vec3<float> nPos;
	// x
	for (size_t x = 0; x < velocities_x_->size_X; ++x)
	{
		for (size_t y = 0; y < velocities_x_->size_Y; ++y)
		{
			for (size_t z = 0; z < velocities_x_->size_Z; ++z)
			{
				if (IsSolidCell(x - 1, y, z) || IsSolidCell(x, y, z))
					continue;
				
				nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} + VelocityXGridOffset;
				math::vec3<float> velocity = SampleVelocity(nPos);
				pos = DenormalizePosition(nPos);
				prevPos = pos - velocity * fixed_time_step;
				nPos = NormalizePosition(prevPos) - VelocityXGridOffset;
				(*velocities_x_temp_)[x][y][z] = Interpolation(velocities_x_.get(), nPos, interpolation_type);
			}
		}
	}
	// y
	for (size_t x = 0; x < velocities_y_->size_X; ++x)
	{
		for (size_t y = 0; y < velocities_y_->size_Y; ++y)
		{
			for (size_t z = 0; z < velocities_y_->size_Z; ++z)
			{
				if (IsSolidCell(x, y - 1, z) || IsSolidCell(x, y, z))
					continue;
				
				nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} + VelocityYGridOffset;
				math::vec3<float> velocity = SampleVelocity(nPos);
				pos = DenormalizePosition(nPos);
				prevPos = pos - velocity * fixed_time_step;
				nPos = NormalizePosition(prevPos) - VelocityYGridOffset;
				(*velocities_y_temp_)[x][y][z] = Interpolation(velocities_y_.get(), nPos, interpolation_type);
			}
		}
	}
	// z
	for (size_t x = 0; x < velocities_z_->size_X; ++x)
	{
		for (size_t y = 0; y < velocities_z_->size_Y; ++y)
		{
			for (size_t z = 0; z < velocities_z_->size_Z; ++z)
			{
				if (IsSolidCell(x, y, z - 1) || IsSolidCell(x, y, z))
					continue;
				
				nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} + VelocityZGridOffset;
				math::vec3<float> velocity = SampleVelocity(nPos);
				pos = DenormalizePosition(nPos);
				prevPos = pos - velocity * fixed_time_step;
				nPos = NormalizePosition(prevPos) - VelocityZGridOffset;
				(*velocities_z_temp_)[x][y][z] = Interpolation(velocities_z_.get(), nPos, interpolation_type);
			}
		}
	}
	
	std::swap(velocities_x_, velocities_x_temp_);
	std::swap(velocities_y_, velocities_y_temp_);
	std::swap(velocities_z_, velocities_z_temp_);
}

void EulerFluidSim::AdvectTemperature()
{
	// TODO: add a temperature smoothing, unrelated to velocity, in order to prevent formation of thin columns
	math::vec3<float> pos;
	math::vec3<float> prevPos;
	math::vec3<float> nPos;
	for (size_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (size_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (size_t z = 0; z < dimensions.z - 1; ++z)
			{
				nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} + ValueGridOffset;
				math::vec3<float> velocity = SampleVelocity(nPos);
				pos = DenormalizePosition(nPos);
				prevPos = pos - velocity * fixed_time_step;
				nPos = NormalizePosition(prevPos) - ValueGridOffset;
				(*cell_temperature_temp_)[x][y][z] = Interpolation(cell_temperature_.get(), nPos, interpolation_type);
			}
		}
	}
	
	std::swap(cell_temperature_, cell_temperature_temp_);
}

void EulerFluidSim::ApplyBuoyancy()
{
	// skip walls, except upper and lower ones
	// k works only with const thermal_expansion_coefficient
	const double k = thermal_expansion_coefficient * gravitational_acceleration_ * fixed_time_step;
	
	for (size_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (size_t y = 0; y < dimensions.y - 1; ++y)
		{
			for (size_t z = 1; z < dimensions.z - 1; ++z)
			{
				double localTemperature = ((*cell_temperature_)[x - 1][y][z] +
										  (*cell_temperature_)[x + 1][y][z] +
										  (*cell_temperature_)[x][y][z - 1] +
										  (*cell_temperature_)[x][y][z + 1] +
										  (*cell_temperature_)[x - 1][y + 1][z] +
										  (*cell_temperature_)[x + 1][y + 1][z] +
										  (*cell_temperature_)[x][y + 1][z - 1] +
										  (*cell_temperature_)[x][y + 1][z + 1]) * 0.125;
				double deltaT = ((*cell_temperature_)[x][y][z] + (*cell_temperature_)[x][y + 1][z]) * 0.5 - localTemperature;
				(*velocities_z_)[x][y + 1][z] += k * deltaT;
			}
		}
	}
}

// Debug draw methods
void EulerFluidSim::DrawGrid(Core::RenderPipeline* renderer, bool persistent) const
{
	if (boundaryOnly)
	{
		auto box = math::vec3<float>{dimensions.x - 2, dimensions.y - 2, dimensions.z - 2} * scale * 0.5f;
		renderer->DebugDrawBox(math::vec3<float>{}, box, grid_color_, persistent);
	}
	else
	{
		math::vec3<float> lineStart = world_origin_offset_;
		math::vec3<float> lineEnd{world_origin_offset_.x, world_origin_offset_.y,world_origin_offset_.z + dimensions.z * scale};
		for (uint64_t x = 0; x <= dimensions.x; ++x)
		{
			lineStart.x = world_origin_offset_.x + x * scale;
			lineEnd.x = world_origin_offset_.x + x * scale;
			for (uint64_t y = 0; y <= dimensions.y; ++y)
			{
				lineStart.y = world_origin_offset_.y + y * scale;
				lineEnd.y = world_origin_offset_.y + y * scale;
				renderer->DebugDrawLine(lineStart, lineEnd, grid_color_, persistent);
			}
			lineStart.y = world_origin_offset_.y;
			// lineEnd is already set
			// lineEnd.y = world_origin_offset_.y + dimensions.y * scale;
			for (uint64_t z = 0; z <= dimensions.z; ++z)
			{
				lineStart.z = world_origin_offset_.z + z * scale;
				lineEnd.z = world_origin_offset_.z + z * scale;
				renderer->DebugDrawLine(lineStart, lineEnd, grid_color_, persistent);
			}
			lineStart.z = world_origin_offset_.z;
			// lineEnd is already set
			// lineEnd.z = world_origin_offset_.z + dimensions.z * scale;
		}
		lineStart.x = world_origin_offset_.x;
		// lineEnd is already set
		// lineEnd.x = world_origin_offset_.x + dimensions.x * scale;
		for (uint64_t y = 0; y <= dimensions.y; ++y)
		{
			lineStart.y = world_origin_offset_.y + y * scale;
			lineEnd.y = world_origin_offset_.y + y * scale;
			for (uint64_t z = 0; z <= dimensions.z; ++z)
			{
				lineStart.z = world_origin_offset_.z + z * scale;
				lineEnd.z = world_origin_offset_.z + z * scale;
				renderer->DebugDrawLine(lineStart, lineEnd, grid_color_, persistent);
			}
		}
	}
}

void EulerFluidSim::DrawBoundaryVelocities(Core::RenderPipeline* renderer, bool persistent) const
{
	math::vec3<float> start;
	double arrow_value;
	for (uint64_t x = 0; x < dimensions.x; ++x)
	{
		for (uint64_t y = 0; y < dimensions.y; ++y)
		{
			for (uint64_t z = 0; z < dimensions.z; ++z)
			{
				start.x = world_origin_offset_.x + scale * x;
				start.y = world_origin_offset_.y + scale * (y + 0.5);
				start.z = world_origin_offset_.z + scale * (z + 0.5);
				arrow_value = (*velocities_x_)[x][y][z];
				renderer->DebugDrawArrow(start, 
										 start + math::vec3<float>{arrow_value * 0.5, 0.0, 0.0},
										  fabsf(arrow_value) * 0.25f,
										 boundary_velocity_color_x_,
										 persistent);
				start.x = world_origin_offset_.x + scale * (x + 0.5);
				start.y = world_origin_offset_.y + scale * y;
				arrow_value = (*velocities_y_)[x][y][z];
				renderer->DebugDrawArrow(start,
										 start + math::vec3<float>{0.0, arrow_value * 0.5, 0.0},
										 fabsf(arrow_value) * 0.25f,
										 boundary_velocity_color_y_,
										 persistent);
				start.y = world_origin_offset_.y + scale * (y + 0.5);
				start.z = world_origin_offset_.z + scale * z;
				arrow_value = (*velocities_z_)[x][y][z];
				renderer->DebugDrawArrow(start,
										 start + math::vec3<float>{0.0, 0.0, arrow_value * 0.5},
										 fabsf(arrow_value) * 0.25f,
										 boundary_velocity_color_z_,
										 persistent);
			}
		}
	}
	
	start.x = world_origin_offset_.x + scale * dimensions.x;
	for (uint64_t y = 1; y < dimensions.y; ++y)
	{
		for (uint64_t z = 1; z < dimensions.z; ++z)
		{
			start.y = world_origin_offset_.y + scale * (y + 0.5);
			start.z = world_origin_offset_.z + scale * (z + 0.5);
			arrow_value = (*velocities_x_)[dimensions.x][y][z];
			renderer->DebugDrawArrow(start,
									 start + math::vec3<float>{arrow_value * 0.5, 0.0, 0.0},
									 fabsf(arrow_value) * 0.25f,
									 boundary_velocity_color_x_,
									 persistent);
		}
	}
	start.y = world_origin_offset_.y + scale * dimensions.y;
	for (uint64_t x = 1; x < dimensions.x; ++x)
	{
		for (uint64_t z = 1; z < dimensions.z; ++z)
		{
			start.x = world_origin_offset_.x + scale * (x + 0.5);
			start.z = world_origin_offset_.z + scale * (z + 0.5);
			arrow_value = (*velocities_y_)[x][dimensions.y][z];
			renderer->DebugDrawArrow(start,
									 start + math::vec3<float>{0.0, arrow_value * 0.5, 0.0},
									 fabsf(arrow_value) * 0.25f,
									 boundary_velocity_color_y_,
									 persistent);
		}
	}
	start.z = world_origin_offset_.z + scale * dimensions.z;
	for (uint64_t x = 0; x < dimensions.x; ++x)
	{
		for (uint64_t y = 0; y < dimensions.y; ++y)
		{
			start.x = world_origin_offset_.x + scale * (x + 0.5);
			start.y = world_origin_offset_.y + scale * (y + 0.5);
			arrow_value = (*velocities_z_)[x][y][dimensions.z];
			renderer->DebugDrawArrow(start,
									 start + math::vec3<float>{0.0, 0.0, arrow_value * 0.5},
									 fabsf(arrow_value) * 0.25f,
									 boundary_velocity_color_z_,
									 persistent);
		}
	}
}

void EulerFluidSim::DrawInterpolatedVelocities(Core::RenderPipeline* renderer, bool persistent) const
{
	math::vec3<float> nPos;
	math::vec3<float> interpolationPos;
	// border cells will have 0 velocity anyway, so we can skip them
	if (cell_subdivision == 1)
	{
		for (uint64_t x = 1; x < dimensions.x - 1; ++x)
		{
			for (uint64_t y = 1; y < dimensions.y - 1; ++y)
			{
				for (uint64_t z = 1; z < dimensions.z - 1; ++z)
				{
					nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} + ValueGridOffset;
					interpolationPos = DenormalizePosition(nPos);
					math::vec3<float> velocity = SampleVelocity(nPos);
					renderer->DebugDrawArrow(interpolationPos,
											 interpolationPos + velocity,
											 velocity.magnitude() * 0.25f,
											 interpolated_velocity_color_,
											 persistent);
				}
			}
		}
	}
	else {
		for (uint64_t x = cell_subdivision; x <= (dimensions.x - 1) * cell_subdivision; ++x)
		{
			for (uint64_t y = cell_subdivision; y <= (dimensions.y - 1) * cell_subdivision; ++y)
			{
				for (uint64_t z = cell_subdivision; z <= (dimensions.z - 1) * cell_subdivision; ++z)
				{
					nPos = math::vec3<float>{
						static_cast<float>(x) / static_cast<float>(cell_subdivision) + 1.0f / static_cast<float>(cell_subdivision + 1),
						static_cast<float>(y) / static_cast<float>(cell_subdivision) + 1.0f / static_cast<float>(cell_subdivision + 1),
						static_cast<float>(z) / static_cast<float>(cell_subdivision) + 1.0f / static_cast<float>(cell_subdivision + 1)};
					interpolationPos = DenormalizePosition(nPos);
					math::vec3<float> velocity = SampleVelocity(nPos) / static_cast<float>(cell_subdivision);
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
/*
void EulerFluidSim::DrawPressure(Core::RenderPipeline* renderer, bool persistent) const
{
	// Skip border cells, since they are not fluid
	const UWorld* world = GetWorld();
	math::vec3<float> boxCenter;
	math::vec3<float> boxExtent = math::vec3<float>{scale, scale, scale} * 0.5;
	
	// calculate min/max for proper colouring
	double maxPressure = std::numeric_limits<double>::lowest();
	double minPressure = std::numeric_limits<double>::max();
	for (uint64_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (uint64_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (uint64_t z = 1; z < dimensions.z - 1; ++z)
			{
				maxPressure = std::max(maxPressure, (*cell_pressure_)[x][y][z]);
				minPressure = std::min(minPressure, (*cell_pressure_)[x][y][z]);
			}
		}
	}
	maxPressure = std::max(maxPressure, 1.0);
	minPressure = std::min(minPressure, -1.0);
	
	for (uint64_t x = 1; x < dimensions.x - 1; ++x)
	{
		boxCenter.x = world_origin_offset_.x + scale * (x + 0.5);
		for (uint64_t y = 1; y < dimensions.y - 1; ++y)
		{
			boxCenter.y = world_origin_offset_.y + scale * (y + 0.5);
			for (uint64_t z = 1; z < dimensions.z - 1; ++z)
			{
				boxCenter.z = world_origin_offset_.z + scale * (z + 0.5);
				double lerp_val = ((*cell_pressure_)[x][y][z] - minPressure) / (maxPressure - minPressure);
				if (isnan(lerp_val))
					lerp_val = 0.5;
				
				DebugDrawSolidBox( boxCenter, boxExtent, pressureColor->GetLinearColorValue(lerp_val).QuantizeFloor(), persistent);
			}
		}
	}
}
void EulerFluidSim::DrawDivergence(Core::RenderPipeline* renderer, bool persistent) const
{
	// Skip border cells, since they are not fluid
	const UWorld* world = GetWorld();
	math::vec3<float> boxCenter;
	math::vec3<float> boxExtent = math::vec3<float>{scale, scale, scale} * 0.5;
	
	// calculate min/max for proper colouring
	double maxDivergence = std::numeric_limits<double>::lowest();
	double minDivergence = std::numeric_limits<double>::max();
	for (uint64_t x = 1; x < dimensions.x - 1; ++x)
	{
		for (uint64_t y = 1; y < dimensions.y - 1; ++y)
		{
			for (uint64_t z = 1; z < dimensions.z - 1; ++z)
			{
				maxDivergence = std::max(maxDivergence, (*cell_pressure_divergence_)[x][y][z]);
				minDivergence = std::min(minDivergence, (*cell_pressure_divergence_)[x][y][z]);
			}
		}
	}
	maxDivergence = std::max(maxDivergence, 1.0);
	minDivergence = std::min(minDivergence, -1.0);
	
	for (uint64_t x = 1; x < dimensions.x - 1; ++x)
	{
		boxCenter.x = world_origin_offset_.x + scale * (x + 0.5);
		for (uint64_t y = 1; y < dimensions.y - 1; ++y)
		{
			boxCenter.y = world_origin_offset_.y + scale * (y + 0.5);
			for (uint64_t z = 1; z < dimensions.z - 1; ++z)
			{
				boxCenter.z = world_origin_offset_.z + scale * (z + 0.5);
				double lerp_val = ((*cell_pressure_divergence_)[x][y][z] - minDivergence) / (maxDivergence - minDivergence);
				if (isnan(lerp_val))
					lerp_val = 0.5;
				
				DebugDrawSolidBox( boxCenter, boxExtent, divergenceColor->GetLinearColorValue(lerp_val).QuantizeFloor(), persistent);
			}
		}
	}
}
void EulerFluidSim::DrawTemperature(Core::RenderPipeline* renderer, bool persistent) const
{
	// Skip border cells, since they are not fluid
	const UWorld* world = GetWorld();
	math::vec3<float> boxCenter;
	math::vec3<float> boxExtent = math::vec3<float>{scale, scale, scale} * 0.5;
	
	// calculate min/max for proper colouring
	double maxTemperature = core_temperature_;
	double minTemperature = ambient_temperature_;
	
	for (uint64_t x = 0; x < dimensions.x; ++x)
	{
		boxCenter.x = world_origin_offset_.x + scale * (x + 0.5);
		for (uint64_t y = 0; y < dimensions.y; ++y)
		{
			boxCenter.y = world_origin_offset_.y + scale * (y + 0.5);
			for (uint64_t z = 0; z < dimensions.z; ++z)
			{
				boxCenter.z = world_origin_offset_.z + scale * (z + 0.5);
				double lerp_val = ((*cell_temperature_)[x][y][z] - minTemperature) / (maxTemperature - minTemperature);
				if (isnan(lerp_val))
					lerp_val = 0.5;
				
				DebugDrawSolidBox( boxCenter, boxExtent, temperatureColor->GetLinearColorValue(lerp_val).QuantizeFloor(), persistent);
			}
		}
	}
}
*/
