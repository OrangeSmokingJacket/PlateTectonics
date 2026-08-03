#pragma once

#include "euler_fluid_sim.hpp"

cudaGraph_t CreateApplyAGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains);

cudaGraph_t CreateApplyMGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains);

cudaGraph_t CreateDotProductR(CudaDispatchBuffers* d_buffers);
cudaGraph_t CreateDotProductD(CudaDispatchBuffers* d_buffers);
cudaGraph_t CreateLengthR(CudaDispatchBuffers* d_buffers);

cudaGraph_t CreateMaxDivergenceGraph(CudaDispatchBuffers* d_buffers,
                                     SimulationDomain* domains,
                                     cublasHandle_t* handle,
                                     cudaStream_t main_stream);

cudaGraph_t CreateXpYP(CudaDispatchBuffers* d_buffers);
cudaGraph_t CreateXpYR(CudaDispatchBuffers* d_buffers);
cudaGraph_t CreateXpYD(CudaDispatchBuffers* d_buffers);

// Temp H2D and D2H transfers
double GetScalarOutput();

void SetSigma(double val);
void SetNewSigma(double val);
void SetAlpha(double val);
void SetBeta(double val);
