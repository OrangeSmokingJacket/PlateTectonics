#pragma once

#include "euler_fluid_sim.hpp"

cudaGraph_t CreateMaxDivergenceGraph(CudaDispatchBuffers* d_buffers);

cudaGraph_t CreatePCG_PreLoop(CudaDispatchBuffers* d_buffers);
cudaGraph_t CreatePCG_MainLoop(CudaDispatchBuffers* d_buffers);

// Temp H2D and D2H transfers
double GetScalarOutput();
double GetErrorOutput();
uint32_t GetIterationsOutput();
