#pragma once

#include "euler_fluid_sim.hpp"

cudaGraph_t CreateApplyAGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains);

cudaGraph_t CreateApplyMGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains);

cudaGraph_t CreateMaxDivergenceGraph(CudaDispatchBuffers* d_buffers,
                                     SimulationDomain* domains,
                                     cublasHandle_t* handle,
                                     cudaStream_t main_stream);

// Temp D2H transfer
double GetMaxDivergence();
