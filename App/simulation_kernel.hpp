#pragma once

#include "euler_fluid_sim.hpp"

cudaGraph_t CreatePCG_Solver(CudaDispatchBuffers* d_buffers);

// Temp H2D and D2H transfers
double GetScalarOutput();
double GetErrorOutput();
uint32_t GetIterationsOutput();
