#pragma once

#include "euler_fluid_sim.hpp"

void ApplyCoefficientsMatrix(
    EulerFluidSim::CudaDispatchBuffers* d_buffers,
    SimulationDomain* domains,
    const std::unique_ptr<SimulationDomain::GridCoefficientsValues> SimulationDomain::* coefficients_matrix,
    const std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* source);

SimulationDomain::GridValues::value_type
RealCellsDotProduct(EulerFluidSim::CudaDispatchBuffers* d_buffers,
                    SimulationDomain* domains,
                    const std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* lhs,
                    const std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* rhs);
