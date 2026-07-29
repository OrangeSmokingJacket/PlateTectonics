#pragma once

#include "euler_fluid_sim.hpp"

cudaGraph_t CreateApplyAGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains);

cudaGraph_t CreateApplyMGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains);
