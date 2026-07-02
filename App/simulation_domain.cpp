#include "simulation_domain.hpp"
#include "euler_fluid_sim.hpp"

SimulationDomain::SimulationDomain(Index3D domain_index)
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
    , domain_index_(domain_index) {
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

    // Initial values like temperature and velocities are setup by parent
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

void SimulationDomain::CalculateCoefficients() {
    static constexpr double k =
        SimulationConstants::fixed_time_step /
        (SimulationConstants::density * SimulationConstants::scale * SimulationConstants::scale);

    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z}))) {
                    continue;
                }

                int flowPosX = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x + 1, y, z}));
                int flowNegX = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x - 1, y, z}));
                int flowPosY = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y + 1, z}));
                int flowNegY = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y - 1, z}));
                int flowPosZ = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z + 1}));
                int flowNegZ = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z - 1}));

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

void SimulationDomain::CalculatePreconditioner() {
    static constexpr double tuning = 0.97;
    static constexpr double epsilon = 1.0e-30;

    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z}))) {
                    continue;
                }

                double ax = math::square(Aplusx(x - 1, y, z) * prec(x - 1, y, z));
                double ay = math::square(Aplusy(x, y - 1, z) * prec(x, y - 1, z));
                double az = math::square(Aplusz(x, y, z - 1) * prec(x, y, z - 1));

                double bx =
                    Aplusx(x - 1, y, z) * (Aplusy(x - 1, y, z) + Aplusz(x - 1, y, z)) * math::square(prec(x - 1, y, z));
                double by =
                    Aplusy(x, y - 1, z) * (Aplusx(x, y - 1, z) + Aplusz(x, y - 1, z)) * math::square(prec(x, y - 1, z));
                double bz =
                    Aplusz(x, y, z - 1) * (Aplusx(x, y, z - 1) + Aplusy(x, y, z - 1)) * math::square(prec(x, y, z - 1));

                double e = Adiag(x, y, z) - ax - ay - az - tuning * (bx + by + bz);
                prec(x, y, z) = 1 / std::sqrt(e + epsilon);
            }
        }
    }
}

void SimulationDomain::CalculateDivergence() {
    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                int flowPosX = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x + 1, y, z}));
                int flowNegX = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x - 1, y, z}));
                int flowPosY = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y + 1, z}));
                int flowNegY = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y - 1, z}));
                int flowPosZ = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z + 1}));
                int flowNegZ = !EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z - 1}));

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
    intermediate_->SetToZero();
    auxiliary_->SetToZero();

    for (size_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                // For now all cels are fluid
                /*
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z}))) {
                    continue;
                }
                */
                double ax = Aplusx(x - 1, y, z) * prec(x - 1, y, z) * intermediate(x - 1, y, z);
                double ay = Aplusy(x, y - 1, z) * prec(x, y - 1, z) * intermediate(x, y - 1, z);
                double az = Aplusz(x, y, z - 1) * prec(x, y, z - 1) * intermediate(x, y, z - 1);

                intermediate(x, y, z) = ((*residual_)[x][y][z] - ax - ay - az) * prec(x, y, z);
            }
        }
    }
    for (int64_t x = dimensions.x - halo_size.x - 1; x > static_cast<int64_t>(halo_size.x) - 1; x--) {
        for (int64_t y = dimensions.y - halo_size.y - 1; y > static_cast<int64_t>(halo_size.y) - 1; y--) {
            for (int64_t z = dimensions.z - halo_size.z - 1; z > static_cast<int64_t>(halo_size.z) - 1; z--) {
                // For now all cels are fluid
                /*
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z}))) {
                    continue;
                }
                */
                double ax = Aplusx(x, y, z) * prec(x, y, z) * aux(x + 1, y, z);
                double ay = Aplusy(x, y, z) * prec(x, y, z) * aux(x, y + 1, z);
                double az = Aplusz(x, y, z) * prec(x, y, z) * aux(x, y, z + 1);

                aux(x, y, z) = (intermediate(x, y, z) - ax - ay - az) * prec(x, y, z);
            }
        }
    }
}

void SimulationDomain::UpdateVelocities() {
    const double k = SimulationConstants::fixed_time_step / (SimulationConstants::density * SimulationConstants::scale);

    for (uint64_t x = halo_size.x; x < dimensions.x - halo_size.x; ++x) {
        for (uint64_t y = halo_size.y; y < dimensions.y - halo_size.y; ++y) {
            for (uint64_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z}))) {
                    (*velocities_x_)[x][y][z] = 0;
                    (*velocities_x_)[x + 1][y][z] = 0;
                    (*velocities_y_)[x][y][z] = 0;
                    (*velocities_y_)[x][y + 1][z] = 0;
                    (*velocities_z_)[x][y][z] = 0;
                    (*velocities_z_)[x][y][z + 1] = 0;
                    continue;
                }
                if (!EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x + 1, y, z}))) {
                    (*velocities_x_)[x + 1][y][z] -= k * ((*cell_pressure_)[x + 1][y][z] - (*cell_pressure_)[x][y][z]);
                }
                if (!EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y + 1, z}))) {
                    (*velocities_y_)[x][y + 1][z] -= k * ((*cell_pressure_)[x][y + 1][z] - (*cell_pressure_)[x][y][z]);
                }
                if (!EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z + 1}))) {
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
        for (size_t y = halo_size.y; y < dimensions.y - halo_size.y - 1; ++y) {
            for (size_t z = halo_size.z; z < dimensions.z - halo_size.z; ++z) {
                double localTemperature =
                    ((*cell_temperature_)[x - 1][y][z] + (*cell_temperature_)[x + 1][y][z] +
                     (*cell_temperature_)[x][y][z - 1] + (*cell_temperature_)[x][y][z + 1] +
                     (*cell_temperature_)[x - 1][y + 1][z] + (*cell_temperature_)[x + 1][y + 1][z] +
                     (*cell_temperature_)[x][y + 1][z - 1] + (*cell_temperature_)[x][y + 1][z + 1]) *
                    0.125;
                double deltaT =
                    ((*cell_temperature_)[x][y][z] + (*cell_temperature_)[x][y + 1][z]) * 0.5 - localTemperature;
                (*velocities_y_)[x][y + 1][z] += k * deltaT;
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
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x - 1, y, z})) ||
                    EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z})))
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
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y - 1, z})) ||
                    EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z})))
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
                if (EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z - 1})) ||
                    EulerFluidSim::IsSolidCell(WorldIndex(domain_index_, Index3D{x, y, z})))
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
