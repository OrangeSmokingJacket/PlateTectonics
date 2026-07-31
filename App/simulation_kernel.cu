#include "simulation_kernel.hpp"
#include <cublas_v2.h>

__device__ int max_divergence_index = 0;
__device__ __managed__ double max_divergence = 0.0;

double GetMaxDivergence() {
    double result;
    cudaMemcpyFromSymbol(&result, max_divergence, sizeof(double));
    return result;
}

constexpr dim3 domain_dimensions{static_cast<uint32_t>(SimulationDomain::dimensions[0]),
                                 static_cast<uint32_t>(SimulationDomain::dimensions[1]),
                                 static_cast<uint32_t>(SimulationDomain::dimensions[2])};

constexpr dim3 halo_size{static_cast<uint32_t>(SimulationDomain::halo_size[0]),
                         static_cast<uint32_t>(SimulationDomain::halo_size[1]),
                         static_cast<uint32_t>(SimulationDomain::halo_size[2])};

__device__ uint32_t GridIndex(uint32_t x, uint32_t y, uint32_t z) {
    return domain_dimensions.y * domain_dimensions.z * x + domain_dimensions.z * y + z;
}

// Dispatched over real_dimensions
template <typename T>
__global__ void ApplyCoefficientsKernel(T* __restrict__ target,
                                        const T* __restrict__ source,
                                        const CoefficientsValues<T>* __restrict__ coefficients) {
    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + halo_size.x;
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + halo_size.y;
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + halo_size.z;

    target[GridIndex(index_x, index_y, index_z)] =
        coefficients[GridIndex(index_x, index_y, index_z)].diag * source[GridIndex(index_x, index_y, index_z)] +
        coefficients[GridIndex(index_x, index_y, index_z)].plusx * source[GridIndex(index_x + 1, index_y, index_z)] +
        coefficients[GridIndex(index_x, index_y, index_z)].plusy * source[GridIndex(index_x, index_y + 1, index_z)] +
        coefficients[GridIndex(index_x, index_y, index_z)].plusz * source[GridIndex(index_x, index_y, index_z + 1)] +
        coefficients[GridIndex(index_x - 1, index_y, index_z)].plusx *
            source[GridIndex(index_x - 1, index_y, index_z)] +
        coefficients[GridIndex(index_x, index_y - 1, index_z)].plusy *
            source[GridIndex(index_x, index_y - 1, index_z)] +
        coefficients[GridIndex(index_x, index_y, index_z - 1)].plusz * source[GridIndex(index_x, index_y, index_z - 1)];
}

/*
template <typename T>
__global__ void MultiplyInplace(T* lhs, const T* rhs, uint32_t max) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < max) {
        lhs[index] *= rhs[index];
    }
}

// CPU side

static constexpr SimulationDomain::GridValues real_grid_stencil = [](){
    SimulationDomain::GridValues result{};
    for(size_t x = 0; x < SimulationDomain::dimensions[0]; ++x) {
        for(size_t y = 0; y < SimulationDomain::dimensions[1]; ++y) {
            for(size_t z = 0; z < SimulationDomain::dimensions[2]; ++z) {
                if (x >= SimulationDomain::halo_size[0] &&
                    y >= SimulationDomain::halo_size[1] &&
                    z >= SimulationDomain::halo_size[2] &&
                    x < SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0] &&
                    y < SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1] &&
                    z < SimulationDomain::dimensions[2] - SimulationDomain::halo_size[2]) {
                        result[x][y][z] = 1.0;
                    } else {
                        result[x][y][z] = 0.0;
                    }
            }
        }
    }
    return result;
}();
*/

using ValueType = SimulationDomain::GridValues::value_type;
static constexpr size_t elements = SimulationDomain::GridValues::cardinality;
static constexpr dim3 block_dimensions{8, 8, 8};
static constexpr dim3 grid_dimensions{SimulationDomain::real_dimensions[0] / block_dimensions.x,
                                      SimulationDomain::real_dimensions[1] / block_dimensions.y,
                                      SimulationDomain::real_dimensions[2] / block_dimensions.z};

void CreateApplySparseMatrixGraph(
    cudaGraph_t* graph,
    CudaDispatchBuffers* d_buffers,
    SimulationDomain* domains,
    size_t domain_index,
    const CudaDispatchBuffers::SB<CoefficientsValues<double>> CudaDispatchBuffers::* d_coefficients_matrix,
    const std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* h_source,
    const CudaDispatchBuffers::SB<double> CudaDispatchBuffers::* d_source) {
    static constexpr size_t vec_size = elements * sizeof(ValueType);

    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    cudaMemcpyAsync((d_buffers->*d_source).GetSegment(domain_index),
                    (domains[domain_index].*h_source)->flat_data(),
                    vec_size,
                    cudaMemcpyHostToDevice,
                    capture_stream);

    ApplyCoefficientsKernel<ValueType><<<grid_dimensions, block_dimensions, 0, capture_stream>>>(
        d_buffers->auxiliary.GetSegment(domain_index),
        (d_buffers->*d_source).GetSegment(domain_index),
        (d_buffers->*d_coefficients_matrix).GetSegment(domain_index));

    cudaMemcpyAsync(domains[domain_index].auxiliary_->flat_data(),
                    d_buffers->auxiliary.GetSegment(domain_index),
                    vec_size,
                    cudaMemcpyDeviceToHost,
                    capture_stream);

    cudaStreamEndCapture(capture_stream, graph);
    cudaStreamDestroy(capture_stream);
}

void
CreateApplyAGraph(cudaGraph_t* graph, CudaDispatchBuffers* d_buffers, SimulationDomain* domains, size_t domain_index) {
    CreateApplySparseMatrixGraph(graph,
                                 d_buffers,
                                 domains,
                                 domain_index,
                                 &CudaDispatchBuffers::coefficients,
                                 &SimulationDomain::search_,
                                 &CudaDispatchBuffers::search);
}

void
CreateApplyMGraph(cudaGraph_t* graph, CudaDispatchBuffers* d_buffers, SimulationDomain* domains, size_t domain_index) {
    CreateApplySparseMatrixGraph(graph,
                                 d_buffers,
                                 domains,
                                 domain_index,
                                 &CudaDispatchBuffers::preconditioner,
                                 &SimulationDomain::residual_,
                                 &CudaDispatchBuffers::residual);
}

cudaGraph_t CreateApplyMatrixGraph(CudaDispatchBuffers* d_buffers,
                                   SimulationDomain* domains,
                                   void (*create_func)(cudaGraph_t* graph,
                                                       CudaDispatchBuffers* d_buffers,
                                                       SimulationDomain* domains,
                                                       size_t domain_index)) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraph_t children[EulerFluidSim::domains];
    cudaGraphNode_t nodes[EulerFluidSim::domains];

    for (size_t i = 0; i < EulerFluidSim::domains; ++i) {
        create_func(&children[i], d_buffers, domains, i);
        cudaGraphAddChildGraphNode(&nodes[i], graph, nullptr, 0, children[i]);
        cudaGraphDestroy(children[i]);
    }

    return graph;
}

cudaGraph_t CreateApplyAGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains) {
    return CreateApplyMatrixGraph(d_buffers, domains, CreateApplyAGraph);
}

cudaGraph_t CreateApplyMGraph(CudaDispatchBuffers* d_buffers, SimulationDomain* domains) {
    return CreateApplyMatrixGraph(d_buffers, domains, CreateApplyMGraph);
}

template <typename T>
__global__ void ExtractVectorValue(const T* vector, const int* index, T* value) {
    *value = vector[*index - 1];
}

cudaGraph_t CreateMaxDivergenceGraph(CudaDispatchBuffers* d_buffers,
                                     SimulationDomain* domains,
                                     cublasHandle_t* handle,
                                     cudaStream_t main_stream) {
    static constexpr size_t vec_size = elements * sizeof(ValueType);
    int* d_idx = nullptr;
    double* d_max_divergence = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_idx), max_divergence_index);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_max_divergence), max_divergence);

    cudaStreamSynchronize(main_stream);
    cudaMemset(d_buffers->divergence.ptr, 0, d_buffers->divergence.allocation_size);
    // cuBLAS may allocate on first call
    cublasIdamax(*handle, elements * SimulationConstants::domains, d_buffers->divergence.ptr, 1, d_idx);

    cudaStreamSynchronize(main_stream);

    cudaGraph_t graph;
    cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeThreadLocal);

    for (size_t i = 0; i < SimulationConstants::domains; ++i) {
        cudaMemcpyAsync(d_buffers->divergence.GetSegment(i),
                        domains[i].cell_pressure_divergence_->flat_data(),
                        vec_size,
                        cudaMemcpyHostToDevice,
                        main_stream);
    }

    cublasIdamax(*handle, elements * SimulationConstants::domains, d_buffers->divergence.ptr, 1, d_idx);
    ExtractVectorValue<ValueType><<<1, 1, 0, main_stream>>>(d_buffers->divergence.ptr, d_idx, d_max_divergence);

    cudaStreamEndCapture(main_stream, &graph);

    return graph;
}
