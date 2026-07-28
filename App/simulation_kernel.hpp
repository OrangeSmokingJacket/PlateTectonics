#pragma once

#include "euler_fluid_sim.hpp"

cudaGraph_t CreateApplyAGraph(EulerFluidSim::CudaDispatchBuffers* d_buffers, SimulationDomain* domains);

cudaGraph_t CreateApplyMGraph(EulerFluidSim::CudaDispatchBuffers* d_buffers, SimulationDomain* domains);
