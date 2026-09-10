#pragma once

#include "euler_fluid_sim.hpp"

void SetVelocityDevicePointers(double* x, double* x_temp, double* y, double* y_temp, double* z, double* z_temp);

cudaGraph_t CreateSwapVelocities();

cudaGraph_t CreateSyncOrthogonal(CudaDispatchBuffers* d_buffers,
                                 const CudaDispatchBuffers::SB<double> CudaDispatchBuffers::* grid);

cudaGraph_t CreateSyncVelocitiesOrthogonal();

cudaGraph_t CreatePCG_Solver(CudaDispatchBuffers* d_buffers);

cudaGraph_t CreateCalculateDivergence(CudaDispatchBuffers* d_buffers);

cudaGraph_t CreateUpdateVelocities(CudaDispatchBuffers* d_buffers);

cudaGraph_t CreateAdvectVelocities();

// Temp H2D and D2H transfers
double* GetVelX();
double* GetVelY();
double* GetVelZ();
double GetScalarOutput();
double GetErrorOutput();
uint32_t GetIterationsOutput();
