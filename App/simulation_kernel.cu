#include "simulation_kernel.hpp"
#include <cublas_v2.h>

__device__ int max_divergence_index = 0;
// max divergence and dot_product
__device__ __managed__ double scalar_output = 0.0;

double GetScalarOutput() {
    double result;
    cudaMemcpyFromSymbol(&result, scalar_output, sizeof(double));
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

static constexpr size_t vec_length = EulerFluidSim::total_cells;
static constexpr size_t dispatch_size = EulerFluidSim::max_pow2_multiple;
static constexpr size_t blocks = vec_length / dispatch_size;

// [0, 1]
__device__ uint32_t CellReality(uint32_t flat_index) {
    // localize to domain
    flat_index %= domain_dimensions.x * domain_dimensions.y * domain_dimensions.z;
    uint32_t x = flat_index / (domain_dimensions.y * domain_dimensions.z);
    flat_index -= x * domain_dimensions.y * domain_dimensions.z;
    uint32_t y = flat_index / domain_dimensions.z;
    // z does not have halo's, so it is skipped
    return x >= halo_size.x && y >= halo_size.y && x < (domain_dimensions.x - halo_size.x) &&
           y < (domain_dimensions.y - halo_size.y);
}

template <typename T>
__device__ void Reduction(T* partial_sum, T* result) {
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (stride * 2) == 0) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = partial_sum[0];
    }
}

template <typename T>
__global__ void RealCellsDotProduct(const T* lhs, const T* rhs, T* result) {
    __shared__ T partial_sum[dispatch_size];

    uint32_t vec_pos = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = lhs[vec_pos] * rhs[vec_pos] * CellReality(vec_pos);
    __syncthreads();

    Reduction<T>(partial_sum, result);
}

template <typename T, bool IsPower2>
__global__ void PartialSum(const T* v, T* result) {
    __shared__ T partial_sum[dispatch_size];

    uint32_t vec_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if constexpr (IsPower2) {
        partial_sum[threadIdx.x] = v[vec_pos];
    } else {
        partial_sum[threadIdx.x] = (vec_pos < vec_length) ? v[vec_pos] : T{};
    }
    __syncthreads();

    Reduction<T>(partial_sum, result);
}

cudaGraph_t CreateDotProduct(CudaDispatchBuffers* d_buffers,
                             const CudaDispatchBuffers::SB<double> CudaDispatchBuffers::* rhs_buffer) {
    double* d_scalar_output = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_scalar_output), scalar_output);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    RealCellsDotProduct<ValueType><<<blocks, dispatch_size, 0, capture_stream>>>(d_buffers->auxiliary.ptr,
                                                                                 (d_buffers->*rhs_buffer).ptr,
                                                                                 d_buffers->reduction_buffer);

    static_assert(std::bit_ceil(blocks) <= 1024);
    PartialSum<ValueType, false>
        <<<1, std::bit_ceil(blocks), 0, capture_stream>>>(d_buffers->reduction_buffer, d_scalar_output);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

cudaGraph_t CreateDotProductR(CudaDispatchBuffers* d_buffers) {
    return CreateDotProduct(d_buffers, &CudaDispatchBuffers::residual);
}

cudaGraph_t CreateDotProductD(CudaDispatchBuffers* d_buffers) {
    return CreateDotProduct(d_buffers, &CudaDispatchBuffers::search);
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
    double* d_scalar_output = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_idx), max_divergence_index);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_scalar_output), scalar_output);

    // cuBLAS may allocate on first call
    cudaStreamSynchronize(main_stream);
    cudaMemset(d_buffers->divergence.ptr, 0, d_buffers->divergence.allocation_size);
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
    ExtractVectorValue<ValueType><<<1, 1, 0, main_stream>>>(d_buffers->divergence.ptr, d_idx, d_scalar_output);

    cudaStreamEndCapture(main_stream, &graph);

    return graph;
}
