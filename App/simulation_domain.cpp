#include "simulation_domain.hpp"
#include "euler_fluid_sim.hpp"

SimulationDomain::SimulationDomain(size_t face_index)
    : cell_pressure_(std::make_unique<GridValues>())
    , cell_temperature_(std::make_unique<GridValues>())
    , cell_temperature_temp_(std::make_unique<GridValues>())
    , velocities_x_(std::make_unique<GridXValues>())
    , velocities_x_temp_(std::make_unique<GridXValues>())
    , velocities_y_(std::make_unique<GridYValues>())
    , velocities_y_temp_(std::make_unique<GridYValues>())
    , velocities_z_(std::make_unique<GridZValues>())
    , velocities_z_temp_(std::make_unique<GridZValues>())
    , coefficients_(std::make_unique<GridCoefficientsValues>())
    , preconditioner_(std::make_unique<GridCoefficientsValues>())
    , cell_pressure_divergence_(std::make_unique<GridValues>())
    , residual_(std::make_unique<GridValues>())
    , auxiliary_(std::make_unique<GridValues>())
    , search_(std::make_unique<GridValues>())
    , face_index_(face_index) {
    cell_pressure_->SetToZero();
    coefficients_->SetToZero();
    preconditioner_->SetToZero();
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

    // Initial values like temperature and velocities are setup by parent
}

// Some useful macros to maintain readability
#define Adiag(x, y, z) (*coefficients_)[x][y][z].diag
#define Aplusx(x, y, z) (*coefficients_)[x][y][z].plusx
#define Aplusy(x, y, z) (*coefficients_)[x][y][z].plusy
#define Aplusz(x, y, z) (*coefficients_)[x][y][z].plusz
#define Mdiag(x, y, z) (*preconditioner_)[x][y][z].diag
#define Mplusx(x, y, z) (*preconditioner_)[x][y][z].plusx
#define Mplusy(x, y, z) (*preconditioner_)[x][y][z].plusy
#define Mplusz(x, y, z) (*preconditioner_)[x][y][z].plusz
#define aux(x, y, z) (*auxiliary_)[x][y][z]
#define search(x, y, z) (*search_)[x][y][z]
#define residual(x, y, z) (*residual_)[x][y][z]

void SimulationDomain::CalculateCoefficients() {
    static constexpr double k =
        SimulationConstants::fixed_time_step /
        (SimulationConstants::density * SimulationConstants::scale * SimulationConstants::scale);

    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                if (EulerFluidSim::IsSolidCell(z)) {
                    continue;
                }

                static constexpr int flowPosX = 1;
                static constexpr int flowNegX = 1;
                static constexpr int flowPosY = 1;
                static constexpr int flowNegY = 1;
                int flowPosZ = !EulerFluidSim::IsSolidCell(z + 1);
                int flowNegZ = !EulerFluidSim::IsSolidCell(z - 1);

                int flowSum = flowPosX + flowNegX + flowPosY + flowNegY + flowPosZ + flowNegZ;

                // Pressure coefficients
                Adiag(x, y, z) = flowSum * k;
                Aplusx(x, y, z) = -flowPosX * k;
                Aplusy(x, y, z) = -flowPosY * k;
                Aplusz(x, y, z) = -flowPosZ * k;
                // Ideally those have to be synced, but because of static and uniform border, it is not necessary
                Aplusx(x - 1, y, z) = -flowNegX * k;
                Aplusy(x, y - 1, z) = -flowNegY * k;
                Aplusz(x, y, z - 1) = -flowNegZ * k;
            }
        }
    }
}

// Incomplete Poisson method, with reciprocals stored
void SimulationDomain::CalculatePreconditioner() {
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                if (EulerFluidSim::IsSolidCell(z)) {
                    continue;
                }

                Mplusx(x, y, z) = (Aplusx(x, y, z) != 0.0) ? -Aplusx(x, y, z) / Adiag(x, y, z) : 0.0;
                Mplusy(x, y, z) = (Aplusy(x, y, z) != 0.0) ? -Aplusy(x, y, z) / Adiag(x, y, z) : 0.0;
                Mplusz(x, y, z) = (Aplusz(x, y, z) != 0.0) ? -Aplusz(x, y, z) / Adiag(x, y, z) : 0.0;
                // Ideally those have to be synced, but because of static and uniform border, it is not necessary
                Mplusx(x - 1, y, z) = (Aplusx(x - 1, y, z) != 0.0) ? -Aplusx(x - 1, y, z) / Adiag(x, y, z) : 0.0;
                Mplusy(x, y - 1, z) = (Aplusy(x, y - 1, z) != 0.0) ? -Aplusy(x, y - 1, z) / Adiag(x, y, z) : 0.0;
                Mplusz(x, y, z - 1) = (Aplusz(x, y, z - 1) != 0.0) ? -Aplusz(x, y, z - 1) / Adiag(x, y, z) : 0.0;

                Mdiag(x, y, z) = 1.0 + math::square(Mplusx(x - 1, y, z)) + math::square(Mplusy(x, y - 1, z)) +
                                 math::square(Mplusz(x, y, z - 1));
            }
        }
    }
}

void SimulationDomain::CalculateDivergence() {
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                static constexpr int flowPosX = 1;
                static constexpr int flowNegX = 1;
                static constexpr int flowPosY = 1;
                static constexpr int flowNegY = 1;
                int flowPosZ = !EulerFluidSim::IsSolidCell(z + 1);
                int flowNegZ = !EulerFluidSim::IsSolidCell(z - 1);

                double velocityX_POS = (*velocities_x_)[x + 1][y][z] * flowPosX;
                double velocityX_NEG = (*velocities_x_)[x][y][z] * flowNegX;
                double velocityY_POS = (*velocities_y_)[x][y + 1][z] * flowPosY;
                double velocityY_NEG = (*velocities_y_)[x][y][z] * flowNegY;
                double velocityZ_POS = (*velocities_z_)[x][y][z + 1] * flowPosZ;
                double velocityZ_NEG = (*velocities_z_)[x][y][z] * flowNegZ;

                double gradientX = velocityX_POS - velocityX_NEG;
                double gradientY = velocityY_POS - velocityY_NEG;
                double gradientZ = velocityZ_POS - velocityZ_NEG;

                (*cell_pressure_divergence_)[x][y][z] =
                    -(gradientX + gradientY + gradientZ) / SimulationConstants::scale;
            }
        }
    }
}

void SimulationDomain::ApplyCoefficients() {
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                aux(x, y, z) = Adiag(x, y, z) * search(x, y, z) + Aplusx(x - 1, y, z) * search(x - 1, y, z) +
                               Aplusx(x, y, z) * search(x + 1, y, z) + Aplusy(x, y - 1, z) * search(x, y - 1, z) +
                               Aplusy(x, y, z) * search(x, y + 1, z) + Aplusz(x, y, z - 1) * search(x, y, z - 1) +
                               Aplusz(x, y, z) * search(x, y, z + 1);
            }
        }
    }
}

void SimulationDomain::ApplyPreconditioner() {
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                aux(x, y, z) = Mdiag(x, y, z) * residual(x, y, z) + Mplusx(x - 1, y, z) * residual(x - 1, y, z) +
                               Mplusx(x, y, z) * residual(x + 1, y, z) + Mplusy(x, y - 1, z) * residual(x, y - 1, z) +
                               Mplusy(x, y, z) * residual(x, y + 1, z) + Mplusz(x, y, z - 1) * residual(x, y, z - 1) +
                               Mplusz(x, y, z) * residual(x, y, z + 1);
            }
        }
    }
}

void SimulationDomain::UpdateVelocities() {
    const double k = SimulationConstants::fixed_time_step / (SimulationConstants::density * SimulationConstants::scale);

    for (uint64_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (uint64_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (uint64_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                if (EulerFluidSim::IsSolidCell(z)) {
                    (*velocities_x_)[x][y][z] = 0;
                    (*velocities_x_)[x + 1][y][z] = 0;
                    (*velocities_y_)[x][y][z] = 0;
                    (*velocities_y_)[x][y + 1][z] = 0;
                    (*velocities_z_)[x][y][z] = 0;
                    (*velocities_z_)[x][y][z + 1] = 0;
                    continue;
                }
                (*velocities_x_)[x + 1][y][z] -= k * ((*cell_pressure_)[x + 1][y][z] - (*cell_pressure_)[x][y][z]);
                (*velocities_y_)[x][y + 1][z] -= k * ((*cell_pressure_)[x][y + 1][z] - (*cell_pressure_)[x][y][z]);
                if (!EulerFluidSim::IsSolidCell(z + 1)) {
                    (*velocities_z_)[x][y][z + 1] -= k * ((*cell_pressure_)[x][y][z + 1] - (*cell_pressure_)[x][y][z]);
                }
            }
        }
    }
}

void SimulationDomain::ApplyBuoyancy(const EulerFluidSim* simulation) {
    // skip walls, except upper and lower ones
    // k works only with const thermal_expansion_coefficient
    const double k = simulation->thermal_expansion_coefficient * simulation->gravitational_acceleration_ *
                     SimulationConstants::fixed_time_step;

    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z - 1; ++z) {
                // TODO: account for non-existent halo cells
                double localTemperature =
                    ((*cell_temperature_)[x - 1][y][z] + (*cell_temperature_)[x + 1][y][z] +
                     (*cell_temperature_)[x][y - 1][z] + (*cell_temperature_)[x][y + 1][z] +
                     (*cell_temperature_)[x - 1][y][z + 1] + (*cell_temperature_)[x + 1][y][z + 1] +
                     (*cell_temperature_)[x][y - 1][z + 1] + (*cell_temperature_)[x][y + 1][z + 1]) *
                    0.125;
                double deltaT =
                    ((*cell_temperature_)[x][y][z] + (*cell_temperature_)[x][y][z + 1]) * 0.5 - localTemperature;
                (*velocities_z_)[x][y][z + 1] += k * deltaT;
            }
        }
    }
}

// TODO: fix sampling issues
void SimulationDomain::AdvectVelocities() {
    // TODO: unified loop?
    // TODO: SampleVelocity actually needed only for 2 axes at a time, since one is always at grid local point
    math::vec3<float> pos;
    math::vec3<float> prevPos;
    math::vec3<float> nPos;
    // x
    for (size_t x = halo_size.x; x < velocities_x_->size_X - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < velocities_x_->size_Y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < velocities_x_->size_Z - halo_size.z; ++z) {
                // solid cells have solid neighbours in x/y directions
                if (EulerFluidSim::IsSolidCell(z))
                    continue;

                nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} +
                       VelocityXGridOffset;
                math::vec3<float> velocity = SampleVelocity(nPos);
                assert(!std::isinf(velocity.x));
                assert(!std::isinf(velocity.y));
                assert(!std::isinf(velocity.z));
                assert(!std::isnan(velocity.x));
                assert(!std::isnan(velocity.y));
                assert(!std::isnan(velocity.z));
                pos = DenormalizePosition(nPos);
                prevPos = pos - velocity * SimulationConstants::fixed_time_step;
                nPos = NormalizePosition(prevPos) - VelocityXGridOffset;
                (*velocities_x_temp_)[x][y][z] =
                    Interpolation(velocities_x_.get(), nPos, SimulationConstants::interpolation_type);
            }
        }
    }
    // y
    for (size_t x = halo_size.x; x < velocities_y_->size_X - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < velocities_y_->size_Y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < velocities_y_->size_Z - halo_size.z; ++z) {
                // solid cells have solid neighbours in x/y directions
                if (EulerFluidSim::IsSolidCell(z))
                    continue;

                nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} +
                       VelocityYGridOffset;
                math::vec3<float> velocity = SampleVelocity(nPos);
                pos = DenormalizePosition(nPos);
                prevPos = pos - velocity * SimulationConstants::fixed_time_step;
                nPos = NormalizePosition(prevPos) - VelocityYGridOffset;
                (*velocities_y_temp_)[x][y][z] =
                    Interpolation(velocities_y_.get(), nPos, SimulationConstants::interpolation_type);
            }
        }
    }
    // z
    for (size_t x = halo_size.x; x < velocities_z_->size_X - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < velocities_z_->size_Y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < velocities_z_->size_Z - halo_size.z; ++z) {
                if (EulerFluidSim::IsSolidCell(z - 1) || EulerFluidSim::IsSolidCell(z))
                    continue;

                nPos = math::vec3<float>{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)} +
                       VelocityZGridOffset;
                math::vec3<float> velocity = SampleVelocity(nPos);
                pos = DenormalizePosition(nPos);
                prevPos = pos - velocity * SimulationConstants::fixed_time_step;
                nPos = NormalizePosition(prevPos) - VelocityZGridOffset;
                (*velocities_z_temp_)[x][y][z] =
                    Interpolation(velocities_z_.get(), nPos, SimulationConstants::interpolation_type);
            }
        }
    }

    std::swap(velocities_x_, velocities_x_temp_);
    std::swap(velocities_y_, velocities_y_temp_);
    std::swap(velocities_z_, velocities_z_temp_);
}

void SimulationDomain::AdvectTemperature() {
    // TODO: add a temperature smoothing, unrelated to velocity, in order to prevent formation of thin columns
    math::vec3<float> pos;
    math::vec3<float> prevPos;
    math::vec3<float> nPos;
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                nPos = math::vec3<float>{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} +
                       ValueGridOffset;
                math::vec3<float> velocity = SampleVelocity(nPos);
                pos = DenormalizePosition(nPos);
                prevPos = pos - velocity * SimulationConstants::fixed_time_step;
                nPos = NormalizePosition(prevPos) - ValueGridOffset;
                (*cell_temperature_temp_)[x][y][z] =
                    Interpolation(cell_temperature_.get(), nPos, SimulationConstants::interpolation_type);
            }
        }
    }

    std::swap(cell_temperature_, cell_temperature_temp_);
}

MinMax SimulationDomain::DivergenceMinMax() const {
    double minDivergence = std::numeric_limits<double>::max();
    double maxDivergence = std::numeric_limits<double>::lowest();
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                minDivergence = std::min(minDivergence, (*cell_pressure_divergence_)[x][y][z]);
                maxDivergence = std::max(maxDivergence, (*cell_pressure_divergence_)[x][y][z]);
            }
        }
    }
    return {minDivergence, maxDivergence};
}

math::vec3<float> SimulationDomain::NormalizePosition(const math::vec3<float>& pos) const {
    return pos / SimulationConstants::scale;
}

math::vec3<float> SimulationDomain::DenormalizePosition(const math::vec3<float>& nPos) const {
    return nPos * SimulationConstants::scale;
}

// TODO: account for non-existent halo cells
math::vec3<float> SimulationDomain::SampleVelocity(const math::vec3<float>& nPos) const {
    math::vec3<float> result;
    result.x = Interpolation(velocities_x_.get(), nPos - VelocityXGridOffset, SimulationConstants::interpolation_type);
    result.y = Interpolation(velocities_y_.get(), nPos - VelocityYGridOffset, SimulationConstants::interpolation_type);
    result.z = Interpolation(velocities_z_.get(), nPos - VelocityZGridOffset, SimulationConstants::interpolation_type);
    return result;
}

double SimulationDomain::RealCellsDotProduct(std::unique_ptr<GridValues> SimulationDomain::* lhs,
                                             std::unique_ptr<GridValues> SimulationDomain::* rhs) const {
    double result = 0.0;
    // skip halo cells, because they are either border or belong to another domain
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                result += (*(this->*lhs))[x][y][z] * (*(this->*rhs))[x][y][z];
            }
        }
    }
    return result;
}

double SimulationDomain::RealCellsLength(std::unique_ptr<GridValues> SimulationDomain::* input) const {
    return std::sqrt(RealCellsDotProduct(input, input));
}
