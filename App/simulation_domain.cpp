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
            for (size_t z = border_size; z < dimensions.z - border_size; ++z) {
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
            for (size_t z = border_size; z < dimensions.z - border_size; ++z) {
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
