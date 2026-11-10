#include "simulation_kernel.hpp"
#include <cuda_gl_interop.h>

#include <type_traits>

// Swap buffers
__device__ __managed__ double* d_temperature = nullptr;
__device__ __managed__ double* d_temperature_temp = nullptr;
__device__ __managed__ double* d_velocity_x = nullptr;
__device__ __managed__ double* d_velocity_x_temp = nullptr;
__device__ __managed__ double* d_velocity_y = nullptr;
__device__ __managed__ double* d_velocity_y_temp = nullptr;
__device__ __managed__ double* d_velocity_z = nullptr;
__device__ __managed__ double* d_velocity_z_temp = nullptr;
// Constants
__device__ __managed__ double d_scalar_output = 0.0;
__device__ __managed__ double d_sigma = 0.0;
__device__ __managed__ double d_new_sigma = 0.0;
__device__ __managed__ double d_alpha = 0.0;
__device__ __managed__ double d_beta = 0.0;
__device__ __managed__ double d_total_error = DBL_MAX;
__device__ __managed__ uint32_t d_loop_iteration = 0;
__device__ __managed__ double d_thermal_expansion_coefficient = 0.0;
__device__ __managed__ double d_gravitational_acceleration = 0.0;

double* GetTemperature() {
    double* result;
    cudaMemcpyFromSymbol(&result, d_temperature, sizeof(double*));
    return result;
}
double* GetVelX() {
    double* result;
    cudaMemcpyFromSymbol(&result, d_velocity_x, sizeof(double*));
    return result;
}
double* GetVelY() {
    double* result;
    cudaMemcpyFromSymbol(&result, d_velocity_y, sizeof(double*));
    return result;
}
double* GetVelZ() {
    double* result;
    cudaMemcpyFromSymbol(&result, d_velocity_z, sizeof(double*));
    return result;
}
double GetScalarOutput() {
    double result;
    cudaMemcpyFromSymbol(&result, d_scalar_output, sizeof(double));
    return result;
}
double GetErrorOutput() {
    double result;
    cudaMemcpyFromSymbol(&result, d_total_error, sizeof(double));
    return result;
}
uint32_t GetIterationsOutput() {
    uint32_t result;
    cudaMemcpyFromSymbol(&result, d_loop_iteration, sizeof(uint32_t));
    return result;
}
void SetThermalExpansionCoefficient(double thermal_expansion_coefficient) {
    cudaMemcpyToSymbol(d_thermal_expansion_coefficient,
                       &thermal_expansion_coefficient,
                       sizeof(d_thermal_expansion_coefficient));
}
void SetGravitationalAcceleration(double gravitational_acceleration) {
    cudaMemcpyToSymbol(d_gravitational_acceleration, &gravitational_acceleration, sizeof(d_gravitational_acceleration));
}

using ValueType = SimulationDomain::GridValues::value_type;
static constexpr size_t elements = SimulationDomain::GridValues::cardinality;
static constexpr dim3 block_dimensions{8, 8, 8};
static constexpr dim3 grid_dimensions{SimulationDomain::real_dimensions[0] / block_dimensions.x,
                                      SimulationDomain::real_dimensions[1] / block_dimensions.y,
                                      SimulationDomain::real_dimensions[2] / block_dimensions.z};
__constant__ static constexpr auto world_origin_offset = EulerFluidSim::world_origin_offset;

template <math::vec3<size_t> GridDimensions = SimulationDomain::dimensions>
__device__ constexpr uint32_t GridIndex(uint32_t x, uint32_t y, uint32_t z) {
    if (x >= GridDimensions[0] || y >= GridDimensions[1] || z >= GridDimensions[2]) {
        cuda::std::terminate();
    }
    return GridDimensions[1] * GridDimensions[2] * x + GridDimensions[2] * y + z;
}

template <math::vec3<size_t> GridDimensions = SimulationDomain::dimensions>
__device__ constexpr uint32_t GridIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    if (x >= GridDimensions[0] || y >= GridDimensions[1] || z >= GridDimensions[2] || w >= EulerFluidSim::domains) {
        cuda::std::terminate();
    }
    return GridDimensions[0] * GridDimensions[1] * GridDimensions[2] * w + GridDimensions[1] * GridDimensions[2] * x +
           GridDimensions[2] * y + z;
}

template <math::vec3<size_t> GridDimensions = SimulationDomain::dimensions>
__device__ constexpr uint32_t GridDomainOffset(uint32_t w) {
    return GridDimensions[0] * GridDimensions[1] * GridDimensions[2] * w;
}

__device__ constexpr uint32_t StaggeredGridXDomainOffset(uint32_t w) {
    return CudaDispatchBuffers::VxSB<ValueType>::elements_padded * w;
}

__device__ constexpr uint32_t StaggeredGridYDomainOffset(uint32_t w) {
    return CudaDispatchBuffers::VySB<ValueType>::elements_padded * w;
}

__device__ constexpr uint32_t StaggeredGridZDomainOffset(uint32_t w) {
    return CudaDispatchBuffers::VzSB<ValueType>::elements_padded * w;
}

__device__ constexpr uint32_t StaggeredGridXIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return StaggeredGridXDomainOffset(w) + GridIndex<SimulationDomain::velocitiesXDimensions>(x, y, z);
}

__device__ constexpr uint32_t StaggeredGridYIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return StaggeredGridYDomainOffset(w) + GridIndex<SimulationDomain::velocitiesYDimensions>(x, y, z);
}

__device__ constexpr uint32_t StaggeredGridZIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return StaggeredGridZDomainOffset(w) + GridIndex<SimulationDomain::velocitiesZDimensions>(x, y, z);
}

template <math::vec3<size_t> GridDimensions>
__device__ uint32_t ClampedGridValue(int32_t x, int32_t y, int32_t z) {
#if not defined NDEBUG
    // In practice, it should not have been called outside 1 cell boundary
    if (x < -1 || y < -1 || z < -1 || x > static_cast<int32_t>(GridDimensions[0]) ||
        y > static_cast<int32_t>(GridDimensions[1]) || z > static_cast<int32_t>(GridDimensions[2])) {
        cuda::std::terminate();
    }
#endif
    x = min(max(x, 0), static_cast<int>(GridDimensions[0]) - 1);
    y = min(max(y, 0), static_cast<int>(GridDimensions[1]) - 1);
    z = min(max(z, 0), static_cast<int>(GridDimensions[2]) - 1);
    return GridIndex<GridDimensions>(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z));
}

// Clamped variants are meant for interpolation, where sample stencil can reach 1 cell outside the grid
// Could be united with regular grid index functions

__device__ uint32_t ClampedGridIndex(int32_t x, int32_t y, int32_t z, uint32_t w) {
    return GridDomainOffset(w) + ClampedGridValue<SimulationDomain::dimensions>(x, y, z);
}

__device__ uint32_t ClampedStaggeredGridXIndex(int32_t x, int32_t y, int32_t z, uint32_t w) {
    return StaggeredGridXDomainOffset(w) + ClampedGridValue<SimulationDomain::velocitiesXDimensions>(x, y, z);
}

__device__ uint32_t ClampedStaggeredGridYIndex(int32_t x, int32_t y, int32_t z, uint32_t w) {
    return StaggeredGridYDomainOffset(w) + ClampedGridValue<SimulationDomain::velocitiesYDimensions>(x, y, z);
}

__device__ uint32_t ClampedStaggeredGridZIndex(int32_t x, int32_t y, int32_t z, uint32_t w) {
    return StaggeredGridZDomainOffset(w) + ClampedGridValue<SimulationDomain::velocitiesZDimensions>(x, y, z);
}

// Dispatched over real_dimensions
template <typename T>
__global__ void ApplyCoefficientsKernel(T* __restrict__ target,
                                        const T* __restrict__ source,
                                        const CoefficientsValues<T>* __restrict__ coefficients) {
    if (blockIdx.z * blockDim.z + threadIdx.z < SimulationDomain::dimensions[2] - SimulationDomain::border_size * 2) {
        uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
        uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::border_size;

        target[GridIndex<>(index_x, index_y, index_z)] =
            coefficients[GridIndex(index_x, index_y, index_z)].diag * source[GridIndex(index_x, index_y, index_z)] +
            coefficients[GridIndex(index_x, index_y, index_z)].plusx *
                source[GridIndex(index_x + 1, index_y, index_z)] +
            coefficients[GridIndex(index_x, index_y, index_z)].plusy *
                source[GridIndex(index_x, index_y + 1, index_z)] +
            coefficients[GridIndex(index_x, index_y, index_z)].plusz *
                source[GridIndex(index_x, index_y, index_z + 1)] +
            coefficients[GridIndex(index_x - 1, index_y, index_z)].plusx *
                source[GridIndex(index_x - 1, index_y, index_z)] +
            coefficients[GridIndex(index_x, index_y - 1, index_z)].plusy *
                source[GridIndex(index_x, index_y - 1, index_z)] +
            coefficients[GridIndex(index_x, index_y, index_z - 1)].plusz *
                source[GridIndex(index_x, index_y, index_z - 1)];
    }
}

void CreateApplySparseMatrixGraph(
    cudaGraph_t* graph,
    CudaDispatchBuffers* d_buffers,
    size_t domain_index,
    const CudaDispatchBuffers::SB<CoefficientsValues<double>> CudaDispatchBuffers::* d_coefficients_matrix,
    const CudaDispatchBuffers::SB<double> CudaDispatchBuffers::* d_source) {
    static constexpr size_t vec_size = elements * sizeof(ValueType);

    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    ApplyCoefficientsKernel<ValueType><<<grid_dimensions, block_dimensions, 0, capture_stream>>>(
        d_buffers->auxiliary.GetSegment(domain_index),
        (d_buffers->*d_source).GetSegment(domain_index),
        (d_buffers->*d_coefficients_matrix).GetSegment(domain_index));

    cudaStreamEndCapture(capture_stream, graph);
    cudaStreamDestroy(capture_stream);
}

void CreateApplyAGraph(cudaGraph_t* graph, CudaDispatchBuffers* d_buffers, size_t domain_index) {
    CreateApplySparseMatrixGraph(graph,
                                 d_buffers,
                                 domain_index,
                                 &CudaDispatchBuffers::coefficients,
                                 &CudaDispatchBuffers::search);
}

void CreateApplyMGraph(cudaGraph_t* graph, CudaDispatchBuffers* d_buffers, size_t domain_index) {
    CreateApplySparseMatrixGraph(graph,
                                 d_buffers,
                                 domain_index,
                                 &CudaDispatchBuffers::preconditioner,
                                 &CudaDispatchBuffers::residual);
}

cudaGraph_t
CreateApplyMatrixGraph(CudaDispatchBuffers* d_buffers,
                       void (*create_func)(cudaGraph_t* graph, CudaDispatchBuffers* d_buffers, size_t domain_index)) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraph_t children[EulerFluidSim::domains];
    cudaGraphNode_t nodes[EulerFluidSim::domains];

    for (size_t i = 0; i < EulerFluidSim::domains; ++i) {
        create_func(&children[i], d_buffers, i);
        cudaGraphAddChildGraphNode(&nodes[i], graph, nullptr, 0, children[i]);
        cudaGraphDestroy(children[i]);
    }

    return graph;
}

cudaGraph_t CreateApplyAGraph(CudaDispatchBuffers* d_buffers) {
    return CreateApplyMatrixGraph(d_buffers, CreateApplyAGraph);
}

cudaGraph_t CreateApplyMGraph(CudaDispatchBuffers* d_buffers) {
    return CreateApplyMatrixGraph(d_buffers, CreateApplyMGraph);
}

static constexpr size_t vec_length = EulerFluidSim::total_cells;
static constexpr size_t dispatch_size = EulerFluidSim::max_pow2_multiple;
static constexpr size_t reduction_buffer_size = vec_length / dispatch_size;
static constexpr size_t blocks = vec_length / dispatch_size;
static constexpr size_t blocks1 = blocks / dispatch_size + (blocks % dispatch_size != 0);

// [0, 1]
__device__ uint32_t CellReality(uint32_t flat_index) {
    // localize to domain
    flat_index %= SimulationDomain::dimensions[0] * SimulationDomain::dimensions[1] * SimulationDomain::dimensions[2];
    uint32_t x = flat_index / (SimulationDomain::dimensions[1] * SimulationDomain::dimensions[2]);
    flat_index -= x * SimulationDomain::dimensions[1] * SimulationDomain::dimensions[2];
    uint32_t y = flat_index / SimulationDomain::dimensions[2];
    // z does not have halo's, so it is skipped
    return x >= SimulationDomain::halo_size[0] && y >= SimulationDomain::halo_size[1] &&
           x < (SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0]) &&
           y < (SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1]);
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
__global__ void PartialSum(const T* v, T* result, size_t vector_length) {
    __shared__ T partial_sum[dispatch_size];

    uint32_t vec_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if constexpr (IsPower2) {
        partial_sum[threadIdx.x] = v[vec_pos];
    } else {
        partial_sum[threadIdx.x] = (vec_pos < vector_length) ? v[vec_pos] : T{};
    }
    __syncthreads();

    Reduction<T>(partial_sum, result);
}

template <typename T, bool IsPower2>
__global__ void AbsMax(const T* v, T* result, size_t vector_length) {
    __shared__ T s_data[dispatch_size];

    uint32_t vec_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if constexpr (IsPower2) {
        s_data[threadIdx.x] = v[vec_pos];
    } else {
        s_data[threadIdx.x] = (vec_pos < vector_length) ? v[vec_pos] : T{};
    }
    __syncthreads();

    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (stride * 2) == 0) {
            s_data[threadIdx.x] = max(abs(s_data[threadIdx.x]), abs(s_data[threadIdx.x + stride]));
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = s_data[0];
    }
}

cudaGraph_t CreateDotProduct(CudaDispatchBuffers* d_buffers,
                             const CudaDispatchBuffers::SB<double> CudaDispatchBuffers::* lhs_buffer,
                             const CudaDispatchBuffers::SB<double> CudaDispatchBuffers::* rhs_buffer,
                             double* d_scalar_output_ptr) {
    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    RealCellsDotProduct<ValueType><<<blocks, dispatch_size, 0, capture_stream>>>((d_buffers->*lhs_buffer).ptr,
                                                                                 (d_buffers->*rhs_buffer).ptr,
                                                                                 d_buffers->reduction_buffer);
    PartialSum<ValueType, false><<<blocks1, dispatch_size, 0, capture_stream>>>(d_buffers->reduction_buffer,
                                                                                d_buffers->reduction_buffer,
                                                                                reduction_buffer_size);
    static_assert(blocks1 <= dispatch_size);
    PartialSum<ValueType, false><<<1, dispatch_size, 0, capture_stream>>>(d_buffers->reduction_buffer,
                                                                          d_scalar_output_ptr,
                                                                          reduction_buffer_size);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

cudaGraph_t CreateDotProductR(CudaDispatchBuffers* d_buffers) {
    double* d_new_sigma_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_new_sigma_ptr), d_new_sigma);
    return CreateDotProduct(d_buffers,
                            &CudaDispatchBuffers::auxiliary,
                            &CudaDispatchBuffers::residual,
                            d_new_sigma_ptr);
}
cudaGraph_t CreateDotProductD(CudaDispatchBuffers* d_buffers) {
    double* d_scalar_output_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_scalar_output_ptr), d_scalar_output);
    return CreateDotProduct(d_buffers,
                            &CudaDispatchBuffers::auxiliary,
                            &CudaDispatchBuffers::search,
                            d_scalar_output_ptr);
}
cudaGraph_t CreateLengthR(CudaDispatchBuffers* d_buffers) {
    double* d_total_error_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_total_error_ptr), d_total_error);
    return CreateDotProduct(d_buffers,
                            &CudaDispatchBuffers::residual,
                            &CudaDispatchBuffers::residual,
                            d_total_error_ptr);
}

template <typename T>
__global__ void ExtractVectorValue(const T* vector, const int* index, T* value) {
    *value = vector[*index - 1];
}

cudaGraph_t CreateMaxDivergenceGraph(CudaDispatchBuffers* d_buffers) {
    double* d_scalar_output_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_scalar_output_ptr), d_scalar_output);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    AbsMax<ValueType, true><<<blocks, dispatch_size, 0, capture_stream>>>(d_buffers->divergence.ptr,
                                                                          d_buffers->reduction_buffer,
                                                                          vec_length);
    AbsMax<ValueType, false><<<blocks1, dispatch_size, 0, capture_stream>>>(d_buffers->reduction_buffer,
                                                                            d_buffers->reduction_buffer,
                                                                            reduction_buffer_size);
    static_assert(blocks1 <= dispatch_size);
    AbsMax<ValueType, false><<<1, dispatch_size, 0, capture_stream>>>(d_buffers->reduction_buffer,
                                                                      d_scalar_output_ptr,
                                                                      reduction_buffer_size);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

enum class AXpY_op : uint8_t { plus, minus };

template <typename T, AXpY_op op>
__global__ void AXpY(const T* __restrict__ x, T* __restrict__ y, const T* a) {
    size_t vec_pos = blockDim.x * blockIdx.x + threadIdx.x;
    if constexpr (op == AXpY_op::plus) {
        y[vec_pos] = *a * x[vec_pos] + y[vec_pos];
    } else {
        y[vec_pos] = -(*a) * x[vec_pos] + y[vec_pos];
    }
}

template <typename T>
__global__ void XpAY(const T* __restrict__ x, T* __restrict__ y, const T* a) {
    size_t vec_pos = blockDim.x * blockIdx.x + threadIdx.x;
    y[vec_pos] = x[vec_pos] + *a * y[vec_pos];
}

cudaGraph_t CreateXpYP(CudaDispatchBuffers* d_buffers) {
    double* d_alpha_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_alpha_ptr), d_alpha);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    AXpY<ValueType, AXpY_op::plus>
        <<<blocks, dispatch_size, 0, capture_stream>>>(d_buffers->search.ptr, d_buffers->pressure.ptr, d_alpha_ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}
cudaGraph_t CreateXpYR(CudaDispatchBuffers* d_buffers) {
    double* d_alpha_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_alpha_ptr), d_alpha);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    AXpY<ValueType, AXpY_op::minus>
        <<<blocks, dispatch_size, 0, capture_stream>>>(d_buffers->auxiliary.ptr, d_buffers->residual.ptr, d_alpha_ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}
cudaGraph_t CreateXpYD(CudaDispatchBuffers* d_buffers) {
    double* d_beta_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_beta_ptr), d_beta);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    XpAY<ValueType>
        <<<blocks, dispatch_size, 0, capture_stream>>>(d_buffers->auxiliary.ptr, d_buffers->search.ptr, d_beta_ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

void SetVelocityDevicePointers(double* temperature,
                               double* temperature_temp,
                               double* x,
                               double* x_temp,
                               double* y,
                               double* y_temp,
                               double* z,
                               double* z_temp) {
    cudaMemcpyToSymbol(d_temperature, &temperature, sizeof(double*));
    cudaMemcpyToSymbol(d_temperature_temp, &temperature_temp, sizeof(double*));
    cudaMemcpyToSymbol(d_velocity_x, &x, sizeof(double*));
    cudaMemcpyToSymbol(d_velocity_x_temp, &x_temp, sizeof(double*));
    cudaMemcpyToSymbol(d_velocity_y, &y, sizeof(double*));
    cudaMemcpyToSymbol(d_velocity_y_temp, &y_temp, sizeof(double*));
    cudaMemcpyToSymbol(d_velocity_z, &z, sizeof(double*));
    cudaMemcpyToSymbol(d_velocity_z_temp, &z_temp, sizeof(double*));
}

__global__ void SwapTemperatureBuffer() {
    double* temp = d_temperature;
    d_temperature = d_temperature_temp;
    d_temperature_temp = temp;
}

__global__ void SwapVelocityBuffers() {
    double* temp = d_velocity_x;
    d_velocity_x = d_velocity_x_temp;
    d_velocity_x_temp = temp;

    temp = d_velocity_y;
    d_velocity_y = d_velocity_y_temp;
    d_velocity_y_temp = temp;

    temp = d_velocity_z;
    d_velocity_z = d_velocity_z_temp;
    d_velocity_z_temp = temp;
}

__device__ double* GetGridSegment(double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::SB<double>::elements * index;
}
__device__ const double* GetGridSegment(const double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::SB<double>::elements * index;
}
__device__ double* GetVelocityXSegment(double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::VxSB<double>::elements_padded * index;
}
__device__ const double* GetVelocityXSegment(const double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::VxSB<double>::elements_padded * index;
}
__device__ double* GetVelocityYSegment(double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::VySB<double>::elements_padded * index;
}
__device__ const double* GetVelocityYSegment(const double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::VySB<double>::elements_padded * index;
}
__device__ double* GetVelocityZSegment(double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::VzSB<double>::elements_padded * index;
}
__device__ const double* GetVelocityZSegment(const double* ptr, uint32_t index) {
    return ptr + CudaDispatchBuffers::VzSB<double>::elements_padded * index;
}

cudaGraph_t CreateSwapVelocities() {
    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);

    SwapVelocityBuffers<<<1, 1, 0, capture_stream>>>();

    cudaStreamEndCapture(capture_stream, &graph);
    return graph;
}

// For W neighbours it is cheaper to look it up, than compute
struct WNeighbour {
    size_t x_plus;
    size_t x_minus;
    size_t y_plus;
    size_t y_minus;
};
// Get neighbouring face in specific direction
__device__ constexpr WNeighbour w_neighbours[] = {WNeighbour{1, 4, 2, 5},
                                                  WNeighbour{2, 5, 0, 3},
                                                  WNeighbour{0, 3, 1, 4},
                                                  WNeighbour{4, 1, 5, 2},
                                                  WNeighbour{5, 2, 3, 0},
                                                  WNeighbour{3, 0, 4, 1}};

// dispatched over real cells
template <typename T>
__global__ void SyncFromXPlus(T* v) {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0];
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = index_y;
    uint32_t n_index_y = SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1] - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_plus;

    v[GridIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        v[GridIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
template <typename T>
__global__ void SyncFromXMinus(T* v) {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = 0;
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = SimulationDomain::dimensions[0] - index_y - 1;
    uint32_t n_index_y = SimulationDomain::halo_size[1] * 2 - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_minus;

    v[GridIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        v[GridIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
template <typename T>
__global__ void SyncFromYPlus(T* v) {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0] - 1;
    uint32_t n_index_y = index_x;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_plus;

    v[GridIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        v[GridIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}
template <typename T>
__global__ void SyncFromYMinus(T* v) {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = 0;
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::halo_size[0] * 2 - 1;
    uint32_t n_index_y = SimulationDomain::dimensions[1] - index_x - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_minus;

    v[GridIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        v[GridIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}

// dispatched over real cells
__global__ void SyncTemperatureFromXPlus() {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0];
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = index_y;
    uint32_t n_index_y = SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1] - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_plus;

    d_temperature[GridIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        d_temperature[GridIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
__global__ void SyncTemperatureFromXMinus() {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = 0;
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = SimulationDomain::dimensions[0] - index_y - 1;
    uint32_t n_index_y = SimulationDomain::halo_size[1] * 2 - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_minus;

    d_temperature[GridIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        d_temperature[GridIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
__global__ void SyncTemperatureFromYPlus() {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = SimulationDomain::dimensions[1] - SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::dimensions[0] - SimulationDomain::halo_size[0] - 1;
    uint32_t n_index_y = index_x;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_plus;

    d_temperature[GridIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        d_temperature[GridIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}
__global__ void SyncTemperatureFromYMinus() {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = 0;
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::halo_size[0] * 2 - 1;
    uint32_t n_index_y = SimulationDomain::dimensions[1] - index_x - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_minus;

    d_temperature[GridIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        d_temperature[GridIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}

// Due to domain layout selection, X velocity will always be filled from Y and vice versa

//! Dispatch with halo.x + 1 because of stagger grid
__global__ void SyncVelocityXFromXPlus() {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = SimulationDomain::velocitiesXDimensions[0] - SimulationDomain::halo_size[0] - 1;
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = index_y;
    uint32_t n_index_y = SimulationDomain::velocitiesYDimensions[1] - SimulationDomain::halo_size[1] - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_plus;

    // grid is oriented the opposite way here
    d_velocity_x[StaggeredGridXIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        -d_velocity_y[StaggeredGridYIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
__global__ void SyncVelocityXFromXMinus() {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = 0;
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = SimulationDomain::velocitiesYDimensions[0] - index_y - 1;
    uint32_t n_index_y = SimulationDomain::halo_size[1] * 2 - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_minus;

    // grid is oriented the opposite way here
    d_velocity_x[StaggeredGridXIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        -d_velocity_y[StaggeredGridYIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
__global__ void SyncVelocityXFromYPlus() {
    //! X dimension is not mult of blockDim
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    if (blockIdx.x * blockDim.x + threadIdx.x <
        SimulationDomain::velocitiesXDimensions[0] - SimulationDomain::halo_size[0] * 2) {
        uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
        uint32_t index_y = SimulationDomain::velocitiesXDimensions[1] - SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
        uint32_t index_w = blockIdx.y;

        uint32_t n_index_x = SimulationDomain::velocitiesYDimensions[0] - SimulationDomain::halo_size[0] - 1;
        uint32_t n_index_y = index_x;
        uint32_t n_index_z = index_z;
        uint32_t n_index_w = w_neighbours[index_w].y_plus;

        d_velocity_x[StaggeredGridXIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
            d_velocity_y[StaggeredGridYIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
    }
}
__global__ void SyncVelocityXFromYMinus() {
    //! X dimension is not mult of blockDim
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    if (blockIdx.x * blockDim.x + threadIdx.x <
        SimulationDomain::velocitiesXDimensions[0] - SimulationDomain::halo_size[0] * 2) {
        uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
        uint32_t index_y = 0;
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
        uint32_t index_w = blockIdx.y;

        uint32_t n_index_x = SimulationDomain::halo_size[0] * 2 - 1;
        uint32_t n_index_y = SimulationDomain::velocitiesYDimensions[1] - index_x - 1;
        uint32_t n_index_z = index_z;
        uint32_t n_index_w = w_neighbours[index_w].y_minus;

        // grid is oriented the opposite way here
        d_velocity_x[StaggeredGridXIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
            -d_velocity_y[StaggeredGridYIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
    }
}

__global__ void SyncVelocityYFromXPlus() {
    //! Y dimension is not mult of blockDim
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    if (blockIdx.y * blockDim.y + threadIdx.y <
        SimulationDomain::velocitiesYDimensions[1] - SimulationDomain::halo_size[1] * 2) {
        uint32_t index_x = SimulationDomain::velocitiesYDimensions[0] - SimulationDomain::halo_size[0];
        uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
        uint32_t index_w = blockIdx.x;

        uint32_t n_index_x = index_y;
        uint32_t n_index_y = SimulationDomain::velocitiesXDimensions[1] - SimulationDomain::halo_size[1] - 1;
        uint32_t n_index_z = index_z;
        uint32_t n_index_w = w_neighbours[index_w].x_plus;

        d_velocity_y[StaggeredGridYIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
            d_velocity_x[StaggeredGridXIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
    }
}
__global__ void SyncVelocityYFromXMinus() {
    //! Y dimension is not mult of blockDim
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    if (blockIdx.y * blockDim.y + threadIdx.y <
        SimulationDomain::velocitiesYDimensions[1] - SimulationDomain::halo_size[1] * 2) {
        uint32_t index_x = 0;
        uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
        uint32_t index_w = blockIdx.x;

        uint32_t n_index_x = SimulationDomain::velocitiesXDimensions[0] - index_y - 1;
        uint32_t n_index_y = SimulationDomain::halo_size[1] * 2 - 1;
        uint32_t n_index_z = index_z;
        uint32_t n_index_w = w_neighbours[index_w].x_minus;

        // grid is oriented the opposite way here
        d_velocity_y[StaggeredGridYIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
            -d_velocity_x[StaggeredGridXIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
    }
}
__global__ void SyncVelocityYFromYPlus() {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = SimulationDomain::velocitiesYDimensions[1] - SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::velocitiesXDimensions[0] - SimulationDomain::halo_size[0] - 1;
    uint32_t n_index_y = index_x;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_plus;

    // grid is oriented the opposite way here
    d_velocity_y[StaggeredGridYIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        -d_velocity_x[StaggeredGridXIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}
//! Dispatch with halo.x + 1 because of stagger grid
__global__ void SyncVelocityYFromYMinus() {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = 0;
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::halo_size[0] * 2;
    uint32_t n_index_y = SimulationDomain::velocitiesXDimensions[1] - index_x - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_minus;

    // grid is oriented the opposite way here
    d_velocity_y[StaggeredGridYIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        -d_velocity_x[StaggeredGridXIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}

// Velocity Z technically should be dispatch over real_cells.z + 1 (which is not fit cleanly with bloc dimensions
// but this is a border anyway and will be set to 0, so it is simply skipped
__global__ void SyncVelocityZFromXPlus() {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = SimulationDomain::velocitiesZDimensions[0] - SimulationDomain::halo_size[0];
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = index_y;
    uint32_t n_index_y = SimulationDomain::velocitiesZDimensions[1] - SimulationDomain::halo_size[1] - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_plus;

    d_velocity_z[StaggeredGridZIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        d_velocity_z[StaggeredGridZIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
__global__ void SyncVelocityZFromXMinus() {
    // blockId.x indicates domain
    // blockID.yz - dispatches over domain face
    // threadIdx.x = halo plane

    uint32_t index_x = 0;
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.x;

    uint32_t n_index_x = SimulationDomain::velocitiesZDimensions[0] - index_y - 1;
    uint32_t n_index_y = SimulationDomain::halo_size[1] * 2 - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].x_minus;

    d_velocity_z[StaggeredGridZIndex(index_x + threadIdx.x, index_y, index_z, index_w)] =
        d_velocity_z[StaggeredGridZIndex(n_index_x, n_index_y - threadIdx.x, n_index_z, n_index_w)];
}
__global__ void SyncVelocityZFromYPlus() {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = SimulationDomain::velocitiesZDimensions[1] - SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::velocitiesZDimensions[0] - SimulationDomain::halo_size[0] - 1;
    uint32_t n_index_y = index_x;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_plus;

    d_velocity_z[StaggeredGridZIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        d_velocity_z[StaggeredGridZIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}
__global__ void SyncVelocityZFromYMinus() {
    // blockId.y indicates domain
    // blockID.xz - dispatches over domain face
    // threadIdx.y = halo plane

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = 0;
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];
    uint32_t index_w = blockIdx.y;

    uint32_t n_index_x = SimulationDomain::halo_size[0] * 2 - 1;
    uint32_t n_index_y = SimulationDomain::velocitiesZDimensions[1] - index_x - 1;
    uint32_t n_index_z = index_z;
    uint32_t n_index_w = w_neighbours[index_w].y_minus;

    d_velocity_z[StaggeredGridZIndex(index_x, index_y + threadIdx.y, index_z, index_w)] =
        d_velocity_z[StaggeredGridZIndex(n_index_x - threadIdx.y, n_index_y, n_index_z, n_index_w)];
}

template <typename T>
cudaGraph_t CreateSyncOrthogonalXPlus(CudaDispatchBuffers* d_buffers,
                                      const CudaDispatchBuffers::SB<T> CudaDispatchBuffers::* grid) {
    static constexpr dim3 blocks{EulerFluidSim::domains,
                                 SimulationDomain::real_dimensions[1] / 8,
                                 SimulationDomain::real_dimensions[2] / 8};
    static constexpr dim3 threads{SimulationDomain::halo_size[0], 8, 8};

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    SyncFromXPlus<T><<<blocks, threads, 0, capture_stream>>>((d_buffers->*grid).ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

template <typename T>
cudaGraph_t CreateSyncOrthogonalXMinus(CudaDispatchBuffers* d_buffers,
                                       const CudaDispatchBuffers::SB<T> CudaDispatchBuffers::* grid) {
    static constexpr dim3 blocks{EulerFluidSim::domains,
                                 SimulationDomain::real_dimensions[1] / 8,
                                 SimulationDomain::real_dimensions[2] / 8};
    static constexpr dim3 threads{SimulationDomain::halo_size[0], 8, 8};

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    SyncFromXMinus<T><<<blocks, threads, 0, capture_stream>>>((d_buffers->*grid).ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

template <typename T>
cudaGraph_t CreateSyncOrthogonalYPlus(CudaDispatchBuffers* d_buffers,
                                      const CudaDispatchBuffers::SB<T> CudaDispatchBuffers::* grid) {
    static constexpr dim3 blocks{SimulationDomain::real_dimensions[0] / 8,
                                 EulerFluidSim::domains,
                                 SimulationDomain::real_dimensions[2] / 8};
    static constexpr dim3 threads{8, SimulationDomain::halo_size[1], 8};

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    SyncFromYPlus<T><<<blocks, threads, 0, capture_stream>>>((d_buffers->*grid).ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

template <typename T>
cudaGraph_t CreateSyncOrthogonalYMinus(CudaDispatchBuffers* d_buffers,
                                       const CudaDispatchBuffers::SB<T> CudaDispatchBuffers::* grid) {
    static constexpr dim3 blocks{SimulationDomain::real_dimensions[0] / 8,
                                 EulerFluidSim::domains,
                                 SimulationDomain::real_dimensions[2] / 8};
    static constexpr dim3 threads{8, SimulationDomain::halo_size[1], 8};

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    SyncFromYMinus<T><<<blocks, threads, 0, capture_stream>>>((d_buffers->*grid).ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

cudaGraph_t CreateSyncOrthogonal(CudaDispatchBuffers* d_buffers,
                                 const CudaDispatchBuffers::SB<double> CudaDispatchBuffers::* grid) {
    static constexpr size_t directions = 4;
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t nodes[directions];

    {
        auto child = CreateSyncOrthogonalXPlus(d_buffers, grid);
        cudaGraphAddChildGraphNode(&nodes[0], graph, nullptr, 0, child);
        cudaGraphDestroy(child);
    }
    {
        auto child = CreateSyncOrthogonalXMinus(d_buffers, grid);
        cudaGraphAddChildGraphNode(&nodes[1], graph, nullptr, 0, child);
        cudaGraphDestroy(child);
    }
    {
        auto child = CreateSyncOrthogonalYPlus(d_buffers, grid);
        cudaGraphAddChildGraphNode(&nodes[2], graph, nullptr, 0, child);
        cudaGraphDestroy(child);
    }
    {
        auto child = CreateSyncOrthogonalYMinus(d_buffers, grid);
        cudaGraphAddChildGraphNode(&nodes[3], graph, nullptr, 0, child);
        cudaGraphDestroy(child);
    }

    return graph;
}

cudaGraph_t CreateSyncTemperatureOrthogonal() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t nodes[4];

    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0], 8, 8};
        params.func = &SyncTemperatureFromXPlus;
        cudaGraphAddKernelNode(&nodes[0], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0], 8, 8};
        params.func = &SyncTemperatureFromXMinus;
        cudaGraphAddKernelNode(&nodes[1], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1], 8};
        params.func = &SyncTemperatureFromYPlus;
        cudaGraphAddKernelNode(&nodes[2], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1], 8};
        params.func = &SyncTemperatureFromYMinus;
        cudaGraphAddKernelNode(&nodes[3], graph, nullptr, 0, &params);
    }

    return graph;
}

cudaGraph_t CreateSyncVelocitiesOrthogonal() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t nodes[12];

    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0] + 1, 8, 8};
        params.func = &SyncVelocityXFromXPlus;
        cudaGraphAddKernelNode(&nodes[0], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0], 8, 8};
        params.func = &SyncVelocityXFromXMinus;
        cudaGraphAddKernelNode(&nodes[1], graph, nullptr, 0, &params);
    }
    {
        // +1 for x dimension because of MAC grid
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8 + 1,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1], 8};
        params.func = &SyncVelocityXFromYPlus;
        cudaGraphAddKernelNode(&nodes[2], graph, nullptr, 0, &params);
    }
    {
        // +1 for x dimension because of MAC grid
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8 + 1,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1], 8};
        params.func = &SyncVelocityXFromYMinus;
        cudaGraphAddKernelNode(&nodes[3], graph, nullptr, 0, &params);
    }

    {
        // +1 for y dimension because of MAC grid
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8 + 1,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0], 8, 8};
        params.func = &SyncVelocityYFromXPlus;
        cudaGraphAddKernelNode(&nodes[4], graph, nullptr, 0, &params);
    }
    {
        // +1 for y dimension because of MAC grid
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8 + 1,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0], 8, 8};
        params.func = &SyncVelocityYFromXMinus;
        cudaGraphAddKernelNode(&nodes[5], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1], 8};
        params.func = &SyncVelocityYFromYPlus;
        cudaGraphAddKernelNode(&nodes[6], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1] + 1, 8};
        params.func = &SyncVelocityYFromYMinus;
        cudaGraphAddKernelNode(&nodes[7], graph, nullptr, 0, &params);
    }

    // Technically there should be +1 on every z direction, but it falls on the boundary and will be set to 0 anyway
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0], 8, 8};
        params.func = &SyncVelocityZFromXPlus;
        cudaGraphAddKernelNode(&nodes[8], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{SimulationDomain::halo_size[0], 8, 8};
        params.func = &SyncVelocityZFromXMinus;
        cudaGraphAddKernelNode(&nodes[9], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1], 8};
        params.func = &SyncVelocityZFromYPlus;
        cudaGraphAddKernelNode(&nodes[10], graph, nullptr, 0, &params);
    }
    {
        cudaKernelNodeParams params{};
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              EulerFluidSim::domains,
                              SimulationDomain::real_dimensions[2] / 8};
        params.blockDim = dim3{8, SimulationDomain::halo_size[1], 8};
        params.func = &SyncVelocityZFromYMinus;
        cudaGraphAddKernelNode(&nodes[11], graph, nullptr, 0, &params);
    }

    return graph;
}

template <typename T>
__global__ void DivideScalars(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ res_out) {
    *res_out = *a / *b;
}

cudaGraph_t CreateCalculateAlpha() {
    double* d_alpha_ptr = nullptr;
    double* d_sigma_ptr = nullptr;
    double* d_scalar_output_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_alpha_ptr), d_alpha);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_sigma_ptr), d_sigma);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_scalar_output_ptr), d_scalar_output);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    DivideScalars<ValueType><<<1, 1, 0, capture_stream>>>(d_sigma_ptr, d_scalar_output_ptr, d_alpha_ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

cudaGraph_t CreateCalculateBeta() {
    double* d_beta_ptr = nullptr;
    double* d_new_sigma_ptr = nullptr;
    double* d_sigma_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_beta_ptr), d_beta);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_new_sigma_ptr), d_new_sigma);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_sigma_ptr), d_sigma);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    DivideScalars<ValueType><<<1, 1, 0, capture_stream>>>(d_new_sigma_ptr, d_sigma_ptr, d_beta_ptr);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

cudaGraph_t CreateUpdateSigma() {
    double* d_new_sigma_ptr = nullptr;
    double* d_sigma_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_new_sigma_ptr), d_new_sigma);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_sigma_ptr), d_sigma);

    cudaGraph_t graph;
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    cudaMemcpyAsync(d_sigma_ptr, d_new_sigma_ptr, sizeof(double), cudaMemcpyDeviceToDevice, capture_stream);

    cudaStreamEndCapture(capture_stream, &graph);
    cudaStreamDestroy(capture_stream);

    return graph;
}

cudaGraph_t CreatePCG_PreLoop(CudaDispatchBuffers* d_buffers) {
    cudaGraph_t pre_loop_part;
    cudaGraphCreate(&pre_loop_part);
    cudaGraphNode_t nodes[7];

    {
        cudaMemsetParams memset_params{};
        memset_params.dst = d_buffers->pressure.ptr;
        memset_params.elementSize = 1;
        memset_params.value = 0;
        memset_params.height = 1;
        memset_params.width = d_buffers->pressure.allocation_size;
        cudaGraphAddMemsetNode(&nodes[0], pre_loop_part, nullptr, 0, &memset_params);

        cudaGraphAddMemcpyNode1D(&nodes[1],
                                 pre_loop_part,
                                 nullptr,
                                 0,
                                 d_buffers->residual.ptr,
                                 d_buffers->divergence.ptr,
                                 d_buffers->residual.allocation_size,
                                 cudaMemcpyDeviceToDevice);

        auto apply_M = CreateApplyMGraph(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[2], pre_loop_part, &nodes[0], 2, apply_M);
        cudaGraphDestroy(apply_M);

        auto sync_halos = CreateSyncOrthogonal(d_buffers, &CudaDispatchBuffers::auxiliary);
        cudaGraphAddChildGraphNode(&nodes[3], pre_loop_part, &nodes[2], 1, sync_halos);
        cudaGraphDestroy(sync_halos);

        auto aux_residual_dot_product = CreateDotProductR(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[4], pre_loop_part, &nodes[2], 1, aux_residual_dot_product);
        cudaGraphDestroy(aux_residual_dot_product);

        auto update_sigma = CreateUpdateSigma();
        // depends on setting beta and halo sync
        cudaGraphAddChildGraphNode(&nodes[5], pre_loop_part, &nodes[4], 1, update_sigma);
        cudaGraphDestroy(update_sigma);

        cudaGraphAddMemcpyNode1D(&nodes[6],
                                 pre_loop_part,
                                 &nodes[3],
                                 1,
                                 d_buffers->search.ptr,
                                 d_buffers->auxiliary.ptr,
                                 d_buffers->search.allocation_size,
                                 cudaMemcpyDeviceToDevice);
    }

    return pre_loop_part;
}

cudaGraph_t CreatePCG_MainLoop_Upper(CudaDispatchBuffers* d_buffers) {
    cudaGraph_t loop_upper_part;
    cudaGraphCreate(&loop_upper_part);
    cudaGraphNode_t nodes[7];

    {
        auto apply_A = CreateApplyAGraph(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[0], loop_upper_part, nullptr, 0, apply_A);
        cudaGraphDestroy(apply_A);

        auto sync_halos = CreateSyncOrthogonal(d_buffers, &CudaDispatchBuffers::auxiliary);
        cudaGraphAddChildGraphNode(&nodes[1], loop_upper_part, &nodes[0], 1, sync_halos);
        cudaGraphDestroy(sync_halos);

        auto aux_search_dot_product = CreateDotProductD(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[3], loop_upper_part, &nodes[0], 1, aux_search_dot_product);
        cudaGraphDestroy(aux_search_dot_product);

        auto set_alpha = CreateCalculateAlpha();
        cudaGraphAddChildGraphNode(&nodes[2], loop_upper_part, &nodes[3], 1, set_alpha);
        cudaGraphDestroy(set_alpha);

        auto search_pressure_AXpY = CreateXpYP(d_buffers);
        // depends on setting alpha
        cudaGraphAddChildGraphNode(&nodes[4], loop_upper_part, &nodes[2], 1, search_pressure_AXpY);
        cudaGraphDestroy(search_pressure_AXpY);

        auto auxiliary_residual_AXpY = CreateXpYR(d_buffers);
        // depends on setting alpha and halo sync
        cudaGraphAddChildGraphNode(&nodes[5], loop_upper_part, &nodes[1], 2, auxiliary_residual_AXpY);
        cudaGraphDestroy(auxiliary_residual_AXpY);

        auto residual_length = CreateLengthR(d_buffers);
        // depends on aux-res update
        cudaGraphAddChildGraphNode(&nodes[6], loop_upper_part, &nodes[5], 1, residual_length);
        cudaGraphDestroy(residual_length);
    }

    return loop_upper_part;
}

cudaGraph_t CreatePCG_MainLoop_Lower(CudaDispatchBuffers* d_buffers) {
    cudaGraph_t loop_lower_part;
    cudaGraphCreate(&loop_lower_part);
    cudaGraphNode_t nodes[6];

    {
        auto apply_M = CreateApplyMGraph(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[0], loop_lower_part, nullptr, 0, apply_M);
        cudaGraphDestroy(apply_M);

        auto sync_halos = CreateSyncOrthogonal(d_buffers, &CudaDispatchBuffers::auxiliary);
        cudaGraphAddChildGraphNode(&nodes[1], loop_lower_part, &nodes[0], 1, sync_halos);
        cudaGraphDestroy(sync_halos);

        auto aux_residual_dot_product = CreateDotProductR(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[3], loop_lower_part, &nodes[0], 1, aux_residual_dot_product);
        cudaGraphDestroy(aux_residual_dot_product);

        auto set_beta = CreateCalculateBeta();
        cudaGraphAddChildGraphNode(&nodes[2], loop_lower_part, &nodes[3], 1, set_beta);
        cudaGraphDestroy(set_beta);

        auto auxiliary_search_XpAY = CreateXpYD(d_buffers);
        // depends on setting beta and halo sync
        cudaGraphAddChildGraphNode(&nodes[4], loop_lower_part, &nodes[1], 2, auxiliary_search_XpAY);
        cudaGraphDestroy(auxiliary_search_XpAY);

        auto update_sigma = CreateUpdateSigma();
        // depends on setting beta and halo sync
        cudaGraphAddChildGraphNode(&nodes[5], loop_lower_part, &nodes[2], 1, update_sigma);
        cudaGraphDestroy(update_sigma);
    }

    return loop_lower_part;
}

__global__ void InitLoopKernel(uint32_t* counter, double* error) {
    *counter = 0;
    *error = DBL_MAX;
}

__global__ void LoopKernel(cudaGraphConditionalHandle handle, uint32_t* counter, const double* error) {
    bool expr =
        ++(*counter) <= SimulationConstants::max_solver_iterations && (*error) > SimulationConstants::solver_tolerance;
    cudaGraphSetConditional(handle, expr);
}

__global__ void EarlyLoopExitKernel(cudaGraphConditionalHandle handle, const double* error) {
    bool expr = (*error) > SimulationConstants::solver_tolerance;
    cudaGraphSetConditional(handle, expr);
}

__global__ void SolverActivationKernel(cudaGraphConditionalHandle handle, const double* max_divergence) {
    bool expr = (*max_divergence) > DBL_EPSILON;
    cudaGraphSetConditional(handle, expr);
}

cudaGraph_t CreatePCG_Solver(CudaDispatchBuffers* d_buffers) {
    uint32_t* d_loop_iteration_ptr = nullptr;
    double* d_total_error_ptr = nullptr;
    double* d_max_divergence_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_loop_iteration_ptr), d_loop_iteration);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_total_error_ptr), d_total_error);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_max_divergence_ptr), d_scalar_output);

    cudaGraph_t graph;
    cudaGraphCreate(&graph);
    cudaGraphNode_t nodes[12];

    // initialize loop variables
    {
        cudaGraphNodeParams init_loop_params{cudaGraphNodeTypeKernel};
        void* init_loop_args[2];
        init_loop_args[0] = &d_loop_iteration_ptr;
        init_loop_args[1] = &d_total_error_ptr;
        init_loop_params.kernel.func = InitLoopKernel;
        init_loop_params.kernel.kernelParams = init_loop_args;
        init_loop_params.kernel.gridDim = dim3(1);
        init_loop_params.kernel.blockDim = dim3(1);
        cudaGraphAddNode(&nodes[0], graph, nullptr, nullptr, 0, &init_loop_params);
    }
    {
        auto max_divergence = CreateMaxDivergenceGraph(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[1], graph, nullptr, 0, max_divergence);
        cudaGraphDestroy(max_divergence);
    }

    cudaGraphConditionalHandle solver_handle;
    cudaGraphConditionalHandleCreate(&solver_handle, graph);

    // set_conditional
    {
        cudaGraphNodeParams eval_conditional_params{cudaGraphNodeTypeKernel};
        void* solver_args[2];
        solver_args[0] = &solver_handle;
        solver_args[1] = &d_max_divergence_ptr;
        eval_conditional_params.kernel.func = SolverActivationKernel;
        eval_conditional_params.kernel.kernelParams = solver_args;
        eval_conditional_params.kernel.gridDim = dim3(1);
        eval_conditional_params.kernel.blockDim = dim3(1);
        cudaGraphAddNode(&nodes[2], graph, &nodes[0], nullptr, 2, &eval_conditional_params);
    }

    cudaGraphNodeParams solver_params{cudaGraphNodeTypeConditional};
    solver_params.conditional.handle = solver_handle;
    solver_params.conditional.type = cudaGraphCondTypeIf;
    solver_params.conditional.size = 1;
    cudaGraphAddNode(&nodes[3], graph, &nodes[2], nullptr, 1, &solver_params);

    cudaGraph_t solver_body = solver_params.conditional.phGraph_out[0];

    cudaGraphConditionalHandle loop_handle;
    cudaGraphConditionalHandleCreate(&loop_handle, solver_body);

    {
        auto pre_loop = CreatePCG_PreLoop(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[4], solver_body, nullptr, 0, pre_loop);
        cudaGraphDestroy(pre_loop);
    }

    // set conditional
    {
        cudaGraphNodeParams eval_conditional_params{cudaGraphNodeTypeKernel};
        void* loop_args[3];
        loop_args[0] = &loop_handle;
        loop_args[1] = &d_loop_iteration_ptr;
        loop_args[2] = &d_total_error_ptr;
        eval_conditional_params.kernel.func = LoopKernel;
        eval_conditional_params.kernel.kernelParams = loop_args;
        eval_conditional_params.kernel.gridDim = dim3(1);
        eval_conditional_params.kernel.blockDim = dim3(1);
        cudaGraphAddNode(&nodes[5], solver_body, &nodes[4], nullptr, 1, &eval_conditional_params);
    }

    cudaGraphNodeParams loop_params{cudaGraphNodeTypeConditional};
    loop_params.conditional.handle = loop_handle;
    loop_params.conditional.type = cudaGraphCondTypeWhile;
    loop_params.conditional.size = 1;
    cudaGraphAddNode(&nodes[6], solver_body, &nodes[5], nullptr, 1, &loop_params);

    cudaGraph_t loop_body = loop_params.conditional.phGraph_out[0];
    {
        auto loop_upper = CreatePCG_MainLoop_Upper(d_buffers);
        cudaGraphAddChildGraphNode(&nodes[7], loop_body, nullptr, 0, loop_upper);
        cudaGraphDestroy(loop_upper);

        cudaGraphConditionalHandle early_exit_handle;
        cudaGraphConditionalHandleCreate(&early_exit_handle, loop_body);

        // set conditional
        {
            cudaGraphNodeParams eval_conditional_params{cudaGraphNodeTypeKernel};
            void* early_exit_args[2];
            early_exit_args[0] = &early_exit_handle;
            early_exit_args[1] = &d_total_error_ptr;
            eval_conditional_params.kernel.func = EarlyLoopExitKernel;
            eval_conditional_params.kernel.kernelParams = early_exit_args;
            eval_conditional_params.kernel.gridDim = dim3(1);
            eval_conditional_params.kernel.blockDim = dim3(1);
            cudaGraphAddNode(&nodes[8], loop_body, &nodes[7], nullptr, 1, &eval_conditional_params);
        }

        cudaGraphNodeParams early_exit_params{cudaGraphNodeTypeConditional};
        early_exit_params.conditional.handle = early_exit_handle;
        early_exit_params.conditional.type = cudaGraphCondTypeIf;
        early_exit_params.conditional.size = 1;
        cudaGraphAddNode(&nodes[9], loop_body, &nodes[8], nullptr, 1, &early_exit_params);

        cudaGraph_t no_exit_body = early_exit_params.conditional.phGraph_out[0];
        {
            auto loop_lower = CreatePCG_MainLoop_Lower(d_buffers);
            cudaGraphAddChildGraphNode(&nodes[10], no_exit_body, nullptr, 0, loop_lower);
            cudaGraphDestroy(loop_lower);
        }

        // set conditional inside a loop
        {
            cudaGraphNodeParams eval_conditional_params{cudaGraphNodeTypeKernel};
            void* loop_args[3];
            loop_args[0] = &loop_handle;
            loop_args[1] = &d_loop_iteration_ptr;
            loop_args[2] = &d_total_error_ptr;
            eval_conditional_params.kernel.func = LoopKernel;
            eval_conditional_params.kernel.kernelParams = loop_args;
            eval_conditional_params.kernel.gridDim = dim3(1);
            eval_conditional_params.kernel.blockDim = dim3(1);
            cudaGraphAddNode(&nodes[11], loop_body, &nodes[9], nullptr, 1, &eval_conditional_params);
        }
    }

    return graph;
}

__device__ bool IsSolidCell(size_t z_index) { return z_index == 0 || z_index + 1 >= SimulationDomain::dimensions[2]; }

// dispatched over real cells
template <typename T>
__global__ void CalculateDivergence(T* divergence, uint32_t segment) {
    const double* vel_x = GetVelocityXSegment(d_velocity_x, segment);
    const double* vel_y = GetVelocityYSegment(d_velocity_y, segment);
    const double* vel_z = GetVelocityZSegment(d_velocity_z, segment);
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];

    if (z >= SimulationDomain::border_size && z < SimulationDomain::dimensions[2] - SimulationDomain::border_size) {
        static constexpr int flowPosX = 1;
        static constexpr int flowNegX = 1;
        static constexpr int flowPosY = 1;
        static constexpr int flowNegY = 1;
        int flowPosZ = !IsSolidCell(z + 1);
        int flowNegZ = !IsSolidCell(z - 1);

        T velocityX_POS = vel_x[GridIndex<SimulationDomain::velocitiesXDimensions>(x + 1, y, z)] * flowPosX;
        T velocityX_NEG = vel_x[GridIndex<SimulationDomain::velocitiesXDimensions>(x, y, z)] * flowNegX;
        T velocityY_POS = vel_y[GridIndex<SimulationDomain::velocitiesYDimensions>(x, y + 1, z)] * flowPosY;
        T velocityY_NEG = vel_y[GridIndex<SimulationDomain::velocitiesYDimensions>(x, y, z)] * flowNegY;
        T velocityZ_POS = vel_z[GridIndex<SimulationDomain::velocitiesZDimensions>(x, y, z + 1)] * flowPosZ;
        T velocityZ_NEG = vel_z[GridIndex<SimulationDomain::velocitiesZDimensions>(x, y, z)] * flowNegZ;

        T gradientX = velocityX_POS - velocityX_NEG;
        T gradientY = velocityY_POS - velocityY_NEG;
        T gradientZ = velocityZ_POS - velocityZ_NEG;

        divergence[GridIndex(x, y, z)] = -(gradientX + gradientY + gradientZ) / SimulationConstants::scale;
    }
}

void CreateCalculateDomainsDivergence(cudaGraph_t* graph, CudaDispatchBuffers* d_buffers, size_t domain_index) {
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);

    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    CalculateDivergence<ValueType>
        <<<grid_dimensions, block_dimensions, 0, capture_stream>>>(d_buffers->divergence.GetSegment(domain_index),
                                                                   domain_index);

    cudaStreamEndCapture(capture_stream, graph);
}

cudaGraph_t CreateCalculateDivergence(CudaDispatchBuffers* d_buffers) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraph_t children[EulerFluidSim::domains];
    cudaGraphNode_t nodes[EulerFluidSim::domains];

    for (size_t i = 0; i < EulerFluidSim::domains; ++i) {
        CreateCalculateDomainsDivergence(&children[i], d_buffers, i);
        cudaGraphAddChildGraphNode(&nodes[i], graph, nullptr, 0, children[i]);
        cudaGraphDestroy(children[i]);
    }

    return graph;
}

__device__ bool IsSolidCell(uint32_t z) { return z == 0 || z + 1 >= SimulationDomain::dimensions[2]; }

// dispatched over real cells
// TODO: cache pressure into shared memory
__global__ void UpdateVelocities(const double* d_pressure, uint32_t w) {
    static constexpr double k =
        SimulationConstants::fixed_time_step / (SimulationConstants::density * SimulationConstants::scale);

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];

    if (IsSolidCell(z)) {
        d_velocity_x[StaggeredGridXIndex(x, y, z, w)] = 0;
        d_velocity_x[StaggeredGridXIndex(x + 1, y, z, w)] = 0;
        d_velocity_y[StaggeredGridYIndex(x, y, z, w)] = 0;
        d_velocity_y[StaggeredGridYIndex(x, y + 1, z, w)] = 0;
        d_velocity_z[StaggeredGridZIndex(x, y, z, w)] = 0;
        d_velocity_z[StaggeredGridZIndex(x, y, z + 1, w)] = 0;
    } else {
        d_velocity_x[StaggeredGridXIndex(x + 1, y, z, w)] -=
            k * (d_pressure[GridIndex(x + 1, y, z, w)] - d_pressure[GridIndex(x, y, z, w)]);
        d_velocity_y[StaggeredGridYIndex(x, y + 1, z, w)] -=
            k * (d_pressure[GridIndex(x, y + 1, z, w)] - d_pressure[GridIndex(x, y, z, w)]);
        if (!IsSolidCell(z + 1)) {
            d_velocity_z[StaggeredGridZIndex(x, y, z + 1, w)] -=
                k * (d_pressure[GridIndex(x, y, z + 1, w)] - d_pressure[GridIndex(x, y, z, w)]);
        }
    }
}

cudaGraph_t CreateUpdateVelocities(CudaDispatchBuffers* d_buffers) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t nodes[EulerFluidSim::domains];

    for (uint32_t domain_index = 0; domain_index < EulerFluidSim::domains; ++domain_index) {
        void* args[2];
        args[0] = &d_buffers->pressure.ptr;
        args[1] = &domain_index;
        cudaKernelNodeParams params{};
        params.gridDim = grid_dimensions;
        params.blockDim = block_dimensions;
        params.func = &UpdateVelocities;
        params.kernelParams = args;
        cudaGraphAddKernelNode(&nodes[domain_index], graph, nullptr, 0, &params);
    }

    return graph;
}

// Works since scaling us uniform
template <typename T>
__device__ T NormalizePosition(T axis_pos) {
    return axis_pos / static_cast<double>(SimulationConstants::scale);
}
template <typename T>
__device__ T DenormalizePosition(T axis_pos) {
    return axis_pos * static_cast<double>(SimulationConstants::scale);
}

template <typename T>
__device__ T lerp(T v0, T v1, T t) {
    return v0 * (T{1} - t) + v1 * t;
}

// TODO: all intermediate values can be done in parallel and cached
template <uint32_t (*GridIndexPtr)(int32_t, int32_t, int32_t, uint32_t)>
__device__ ValueType
TrilinearInterpolation(const ValueType* grid, ValueType pos_x, ValueType pos_y, ValueType pos_z, uint32_t w) {
    int32_t x = static_cast<int32_t>(floor(pos_x));
    int32_t y = static_cast<int32_t>(floor(pos_y));
    int32_t z = static_cast<int32_t>(floor(pos_z));

    ValueType fraction_x = pos_x - static_cast<ValueType>(x);
    ValueType fraction_y = pos_y - static_cast<ValueType>(y);
    ValueType fraction_z = pos_z - static_cast<ValueType>(z);

    // Actual interpolation
    ValueType c00 = lerp(grid[GridIndexPtr(x, y, z, w)], grid[GridIndexPtr(x + 1, y, z, w)], fraction_x);
    ValueType c01 = lerp(grid[GridIndexPtr(x, y, z + 1, w)], grid[GridIndexPtr(x + 1, y, z + 1, w)], fraction_x);
    ValueType c10 = lerp(grid[GridIndexPtr(x, y + 1, z, w)], grid[GridIndexPtr(x + 1, y + 1, z, w)], fraction_x);
    ValueType c11 =
        lerp(grid[GridIndexPtr(x, y + 1, z + 1, w)], grid[GridIndexPtr(x + 1, y + 1, z + 1, w)], fraction_x);

    ValueType c0 = lerp(c00, c10, fraction_y);
    ValueType c1 = lerp(c01, c11, fraction_y);

    return lerp(c0, c1, fraction_z);
}

template <uint32_t (*GridIndexPtr)(int32_t, int32_t, int32_t, uint32_t)>
__device__ ValueType TrilinearInterpolation(const ValueType* grid, math::vec3<ValueType> pos, uint32_t w) {
    return TrilinearInterpolation<GridIndexPtr>(grid, pos[0], pos[1], pos[2], w);
}

// dispatched over real cells
__global__ void AdvectTemperature(uint32_t w) {
    if (blockIdx.z * blockDim.z + threadIdx.z < SimulationDomain::dimensions[2] - SimulationDomain::border_size * 2) {
        uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
        uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::border_size;

        math::vec3<ValueType> n_pos;
        math::vec3<ValueType> prev_pos;
        n_pos[0] = static_cast<ValueType>(index_x) + ValueGridOffset[0];
        n_pos[1] = static_cast<ValueType>(index_y) + ValueGridOffset[1];
        n_pos[2] = static_cast<ValueType>(index_z) + ValueGridOffset[2];

        math::vec3<ValueType> velocity;
        velocity[0] = TrilinearInterpolation<&ClampedStaggeredGridXIndex>(d_velocity_x,
                                                                          n_pos[0] - VelocityXGridOffset[0],
                                                                          n_pos[1] - VelocityXGridOffset[1],
                                                                          n_pos[2] - VelocityXGridOffset[2],
                                                                          w);
        velocity[1] = TrilinearInterpolation<&ClampedStaggeredGridYIndex>(d_velocity_y,
                                                                          n_pos[0] - VelocityYGridOffset[0],
                                                                          n_pos[1] - VelocityYGridOffset[1],
                                                                          n_pos[2] - VelocityYGridOffset[2],
                                                                          w);
        velocity[2] = TrilinearInterpolation<&ClampedStaggeredGridZIndex>(d_velocity_z,
                                                                          n_pos[0] - VelocityZGridOffset[0],
                                                                          n_pos[1] - VelocityZGridOffset[1],
                                                                          n_pos[2] - VelocityZGridOffset[2],
                                                                          w);

        for (size_t i = 0; i < 3; ++i) {
            prev_pos[i] = DenormalizePosition(n_pos[i]) - velocity[i] * SimulationConstants::fixed_time_step;
            n_pos[i] = NormalizePosition(prev_pos[i]) - ValueGridOffset[i];
        }

        d_temperature_temp[GridIndex(index_x, index_y, index_z, w)] =
            TrilinearInterpolation<&ClampedGridIndex>(d_temperature, n_pos[0], n_pos[1], n_pos[2], w);
    }
}

// dispatched over real cells
__global__ void AdvectVelocitiesX(uint32_t w) {
    if (blockIdx.x * blockDim.x + threadIdx.x <
            SimulationDomain::velocitiesXDimensions[0] - SimulationDomain::halo_size[0] * 2 &&
        blockIdx.z * blockDim.z + threadIdx.z <
            SimulationDomain::velocitiesXDimensions[2] - SimulationDomain::border_size * 2) {
        uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
        uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::border_size;

        math::vec3<ValueType> n_pos;
        math::vec3<ValueType> prev_pos;
        n_pos[0] = static_cast<ValueType>(index_x) + VelocityXGridOffset[0];
        n_pos[1] = static_cast<ValueType>(index_y) + VelocityXGridOffset[1];
        n_pos[2] = static_cast<ValueType>(index_z) + VelocityXGridOffset[2];

        math::vec3<ValueType> velocity;
        velocity[0] = TrilinearInterpolation<&ClampedStaggeredGridXIndex>(d_velocity_x,
                                                                          n_pos[0] - VelocityXGridOffset[0],
                                                                          n_pos[1] - VelocityXGridOffset[1],
                                                                          n_pos[2] - VelocityXGridOffset[2],
                                                                          w);
        velocity[1] = TrilinearInterpolation<&ClampedStaggeredGridYIndex>(d_velocity_y,
                                                                          n_pos[0] - VelocityYGridOffset[0],
                                                                          n_pos[1] - VelocityYGridOffset[1],
                                                                          n_pos[2] - VelocityYGridOffset[2],
                                                                          w);
        velocity[2] = TrilinearInterpolation<&ClampedStaggeredGridZIndex>(d_velocity_z,
                                                                          n_pos[0] - VelocityZGridOffset[0],
                                                                          n_pos[1] - VelocityZGridOffset[1],
                                                                          n_pos[2] - VelocityZGridOffset[2],
                                                                          w);

        for (size_t i = 0; i < 3; ++i) {
            prev_pos[i] = DenormalizePosition(n_pos[i]) - velocity[i] * SimulationConstants::fixed_time_step;
            n_pos[i] = NormalizePosition(prev_pos[i]) - VelocityXGridOffset[i];
        }

        d_velocity_x_temp[StaggeredGridXIndex(index_x, index_y, index_z, w)] =
            TrilinearInterpolation<&ClampedStaggeredGridXIndex>(d_velocity_x, n_pos[0], n_pos[1], n_pos[2], w);
    }
}

// dispatched over real cells
__global__ void AdvectVelocitiesY(uint32_t w) {
    if (blockIdx.y * blockDim.y + threadIdx.y <
            SimulationDomain::velocitiesYDimensions[1] - SimulationDomain::halo_size[1] * 2 &&
        blockIdx.z * blockDim.z + threadIdx.z <
            SimulationDomain::velocitiesYDimensions[2] - SimulationDomain::border_size * 2) {
        uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
        uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::border_size;

        math::vec3<ValueType> n_pos;
        math::vec3<ValueType> prev_pos;
        n_pos[0] = static_cast<ValueType>(index_x) + VelocityYGridOffset[0];
        n_pos[1] = static_cast<ValueType>(index_y) + VelocityYGridOffset[1];
        n_pos[2] = static_cast<ValueType>(index_z) + VelocityYGridOffset[2];

        math::vec3<ValueType> velocity;
        velocity[0] = TrilinearInterpolation<&ClampedStaggeredGridXIndex>(d_velocity_x,
                                                                          n_pos[0] - VelocityXGridOffset[0],
                                                                          n_pos[1] - VelocityXGridOffset[1],
                                                                          n_pos[2] - VelocityXGridOffset[2],
                                                                          w);
        velocity[1] = TrilinearInterpolation<&ClampedStaggeredGridYIndex>(d_velocity_y,
                                                                          n_pos[0] - VelocityYGridOffset[0],
                                                                          n_pos[1] - VelocityYGridOffset[1],
                                                                          n_pos[2] - VelocityYGridOffset[2],
                                                                          w);
        velocity[2] = TrilinearInterpolation<&ClampedStaggeredGridZIndex>(d_velocity_z,
                                                                          n_pos[0] - VelocityZGridOffset[0],
                                                                          n_pos[1] - VelocityZGridOffset[1],
                                                                          n_pos[2] - VelocityZGridOffset[2],
                                                                          w);

        for (size_t i = 0; i < 3; ++i) {
            prev_pos[i] = DenormalizePosition(n_pos[i]) - velocity[i] * SimulationConstants::fixed_time_step;
            n_pos[i] = NormalizePosition(prev_pos[i]) - VelocityYGridOffset[i];
        }

        d_velocity_y_temp[StaggeredGridYIndex(index_x, index_y, index_z, w)] =
            TrilinearInterpolation<&ClampedStaggeredGridYIndex>(d_velocity_y, n_pos[0], n_pos[1], n_pos[2], w);
    }
}

// dispatched over real cells
__global__ void AdvectVelocitiesZ(uint32_t w) {
    if (blockIdx.z * blockDim.z + threadIdx.z <
        SimulationDomain::velocitiesZDimensions[2] - SimulationDomain::border_size * 2) {
        uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
        uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
        uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::border_size;

        math::vec3<ValueType> n_pos;
        math::vec3<ValueType> prev_pos;
        n_pos[0] = static_cast<ValueType>(index_x) + VelocityZGridOffset[0];
        n_pos[1] = static_cast<ValueType>(index_y) + VelocityZGridOffset[1];
        n_pos[2] = static_cast<ValueType>(index_z) + VelocityZGridOffset[2];

        math::vec3<ValueType> velocity;
        velocity[0] = TrilinearInterpolation<&ClampedStaggeredGridXIndex>(d_velocity_x,
                                                                          n_pos[0] - VelocityXGridOffset[0],
                                                                          n_pos[1] - VelocityXGridOffset[1],
                                                                          n_pos[2] - VelocityXGridOffset[2],
                                                                          w);
        velocity[1] = TrilinearInterpolation<&ClampedStaggeredGridYIndex>(d_velocity_y,
                                                                          n_pos[0] - VelocityYGridOffset[0],
                                                                          n_pos[1] - VelocityYGridOffset[1],
                                                                          n_pos[2] - VelocityYGridOffset[2],
                                                                          w);
        velocity[2] = TrilinearInterpolation<&ClampedStaggeredGridZIndex>(d_velocity_z,
                                                                          n_pos[0] - VelocityZGridOffset[0],
                                                                          n_pos[1] - VelocityZGridOffset[1],
                                                                          n_pos[2] - VelocityZGridOffset[2],
                                                                          w);

        for (size_t i = 0; i < 3; ++i) {
            prev_pos[i] = DenormalizePosition(n_pos[i]) - velocity[i] * SimulationConstants::fixed_time_step;
            n_pos[i] = NormalizePosition(prev_pos[i]) - VelocityZGridOffset[i];
        }

        d_velocity_z_temp[StaggeredGridZIndex(index_x, index_y, index_z, w)] =
            TrilinearInterpolation<&ClampedStaggeredGridZIndex>(d_velocity_z, n_pos[0], n_pos[1], n_pos[2], w);
    }
}

cudaGraph_t CreateAdvectTemperature() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t nodes[EulerFluidSim::domains + 1];

    for (uint32_t domain_index = 0; domain_index < EulerFluidSim::domains; ++domain_index) {
        void* args[1];
        args[0] = &domain_index;
        cudaKernelNodeParams params{};
        params.blockDim = block_dimensions;
        params.kernelParams = args;
        params.func = &AdvectTemperature;
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        cudaGraphAddKernelNode(&nodes[domain_index], graph, nullptr, 0, &params);
    }

    cudaKernelNodeParams params{};
    params.gridDim = 1;
    params.blockDim = 1;
    params.kernelParams = nullptr;
    params.func = &SwapTemperatureBuffer;

    cudaGraphAddKernelNode(&nodes[EulerFluidSim::domains], graph, &nodes[0], EulerFluidSim::domains, &params);

    return graph;
}

cudaGraph_t CreateAdvectVelocities() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t nodes[EulerFluidSim::domains * 3 + 1];

    for (uint32_t domain_index = 0; domain_index < EulerFluidSim::domains; ++domain_index) {
        void* args[1];
        args[0] = &domain_index;
        cudaKernelNodeParams params{};
        params.blockDim = block_dimensions;
        params.kernelParams = args;

        {
            params.func = &AdvectVelocitiesX;
            params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8 + 1,
                                  SimulationDomain::real_dimensions[1] / 8,
                                  SimulationDomain::real_dimensions[2] / 8};
            cudaGraphAddKernelNode(&nodes[domain_index], graph, nullptr, 0, &params);
        }
        {
            params.func = &AdvectVelocitiesY;
            params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                                  SimulationDomain::real_dimensions[1] / 8 + 1,
                                  SimulationDomain::real_dimensions[2] / 8};
            cudaGraphAddKernelNode(&nodes[domain_index + EulerFluidSim::domains], graph, nullptr, 0, &params);
        }
        {
            params.func = &AdvectVelocitiesZ;
            params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                                  SimulationDomain::real_dimensions[1] / 8,
                                  SimulationDomain::real_dimensions[2] / 8 + 1};
            cudaGraphAddKernelNode(&nodes[domain_index + EulerFluidSim::domains * 2], graph, nullptr, 0, &params);
        }
    }

    cudaKernelNodeParams params{};
    params.gridDim = 1;
    params.blockDim = 1;
    params.kernelParams = nullptr;
    params.func = &SwapVelocityBuffers;

    cudaGraphAddKernelNode(&nodes[EulerFluidSim::domains * 3], graph, &nodes[0], EulerFluidSim::domains * 3, &params);

    return graph;
}

__global__ void ApplyBuoyancy(uint32_t w) {
    // k works only with const thermal_expansion_coefficient
    const double k =
        d_thermal_expansion_coefficient * d_gravitational_acceleration * SimulationConstants::fixed_time_step;

    uint32_t index_x = blockIdx.x * blockDim.x + threadIdx.x + SimulationDomain::halo_size[0];
    uint32_t index_y = blockIdx.y * blockDim.y + threadIdx.y + SimulationDomain::halo_size[1];
    uint32_t index_z = blockIdx.z * blockDim.z + threadIdx.z + SimulationDomain::halo_size[2];

    if (index_z + SimulationDomain::border_size < SimulationDomain::dimensions[2]) {
        double localTemperature = (d_temperature[GridIndex(index_x - 1, index_y, index_z, w)] +
                                   d_temperature[GridIndex(index_x + 1, index_y, index_z, w)] +
                                   d_temperature[GridIndex(index_x, index_y - 1, index_z, w)] +
                                   d_temperature[GridIndex(index_x, index_y + 1, index_z, w)] +
                                   d_temperature[GridIndex(index_x - 1, index_y, index_z + 1, w)] +
                                   d_temperature[GridIndex(index_x + 1, index_y, index_z + 1, w)] +
                                   d_temperature[GridIndex(index_x, index_y - 1, index_z + 1, w)] +
                                   d_temperature[GridIndex(index_x, index_y + 1, index_z + 1, w)]) *
                                  0.125;
        double deltaT = (d_temperature[GridIndex(index_x, index_y, index_z, w)] +
                         d_temperature[GridIndex(index_x, index_y, index_z + 1, w)]) *
                            0.5 -
                        localTemperature;
        d_velocity_z[StaggeredGridZIndex(index_x, index_y, index_z + 1, w)] += k * deltaT;
    }
}

cudaGraph_t CreateApplyBuoyancy() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t nodes[EulerFluidSim::domains];

    for (uint32_t domain_index = 0; domain_index < EulerFluidSim::domains; ++domain_index) {
        void* args[1];
        args[0] = &domain_index;
        cudaKernelNodeParams params{};
        params.blockDim = block_dimensions;
        params.kernelParams = args;
        params.func = &ApplyBuoyancy;
        params.gridDim = dim3{SimulationDomain::real_dimensions[0] / 8,
                              SimulationDomain::real_dimensions[1] / 8,
                              SimulationDomain::real_dimensions[2] / 8};
        cudaGraphAddKernelNode(&nodes[domain_index], graph, nullptr, 0, &params);
    }

    return graph;
}

struct CoordinateSystem {
    size_t x;
    size_t y;
    size_t z;
    bool inverse;
};

__device__ constexpr math::vec3<float> ConvertToFaceCS(const CoordinateSystem& cs, const math::vec3<float>& vec) {
    // it is a reverse transformation
    math::vec3<float> result{vec[2 - cs.z], vec[2 - cs.y], vec[2 - cs.x]};
    return cs.inverse ? -result : result;
}

// Transforms coordinate system for each face
//! Warning: 3-5 faces are in reverse handedness
__device__ constexpr cuda::std::array<CoordinateSystem, 6> cube_face_cs{
    CoordinateSystem{.x = 0, .y = 1, .z = 2, .inverse = false},
    CoordinateSystem{.x = 1, .y = 2, .z = 0, .inverse = false},
    CoordinateSystem{.x = 2, .y = 0, .z = 1, .inverse = false},
    CoordinateSystem{.x = 0, .y = 1, .z = 2, .inverse = true},
    CoordinateSystem{.x = 1, .y = 2, .z = 0, .inverse = true},
    CoordinateSystem{.x = 2, .y = 0, .z = 1, .inverse = true}};

__device__ math::vec3<float> ToWorldCoordinates(size_t face_index, const math::vec3<float>& pos) {
    math::vec3<float> result{pos.x, pos.y, world_origin_offset[2]};
    result /= result.magnitude() / (EulerFluidSim::inner_sphere_radius + pos.z);
    return ConvertToFaceCS(cube_face_cs[face_index], result);
}

__device__ void CreateVelocityArrow(math::vec3<float>* out,
                                    uint32_t arr_index,
                                    math::vec3<float> start,
                                    math::vec3<float> end,
                                    float head_size) {
    auto direction = end - start;
    direction.normalize_safe();
    // best_orthogonal causes flickering when plain is changing
    // weighted average of 2 smallest axis will help, but it does have edge cases where arrow is almost axis aligned
    auto orth = math::best_orthogonal(direction);

    out[arr_index] = start;
    out[arr_index + 1] = end;

    out[arr_index + 2] = end;
    out[arr_index + 3] = end + (direction * -head_size) + (orth * head_size);

    out[arr_index + 4] = end;
    out[arr_index + 5] = end + (direction * -head_size) + (orth * -head_size);
}

__global__ void
FillInterpolatedVelocities(math::vec3<float>* out, uint32_t interpolation_points, uint32_t w, float head_size) {
    const uint32_t sample_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t sample_y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t sample_z = blockIdx.z * blockDim.z + threadIdx.z;

    const math::vec3<uint32_t> cell{sample_x / interpolation_points,
                                    sample_y / interpolation_points,
                                    sample_z / interpolation_points};
    const math::vec3<uint32_t> sub_sample{sample_x % interpolation_points,
                                          sample_y % interpolation_points,
                                          sample_z % interpolation_points};

    const float sample_stride = 1.0f / static_cast<float>(interpolation_points);
    const float sample_offset = sample_stride * 0.5f;

    // result array: [cell][sub-sample][6 vertices]
    auto arr_start_index =
        GridIndex<SimulationDomain::real_dimensions>(cell.x, cell.y, cell.z, w) * 6 * math::cube(interpolation_points);
    arr_start_index += 6 * (interpolation_points * interpolation_points * sub_sample.x +
                            interpolation_points * sub_sample.y + sub_sample.z);

    constexpr auto halo = static_cast<math::vec3<ValueType>>(SimulationDomain::halo_size);
    auto n_pos = static_cast<math::vec3<ValueType>>(cell) + halo;
    n_pos[0] += sample_offset + static_cast<float>(sub_sample.x) * sample_stride;
    n_pos[1] += sample_offset + static_cast<float>(sub_sample.y) * sample_stride;
    n_pos[2] += sample_offset + static_cast<float>(sub_sample.z) * sample_stride;

    math::vec3<ValueType> velocity;
    velocity.x = TrilinearInterpolation<&ClampedStaggeredGridXIndex>(
        d_velocity_x,
        n_pos - static_cast<math::vec3<ValueType>>(VelocityXGridOffset),
        w);
    velocity.y = TrilinearInterpolation<&ClampedStaggeredGridYIndex>(
        d_velocity_y,
        n_pos - static_cast<math::vec3<ValueType>>(VelocityYGridOffset),
        w);
    velocity.z = TrilinearInterpolation<&ClampedStaggeredGridZIndex>(
        d_velocity_z,
        n_pos - static_cast<math::vec3<ValueType>>(VelocityZGridOffset),
        w);
    velocity /= static_cast<ValueType>(interpolation_points);

    constexpr auto offset =
        math::vec3<float>{EulerFluidSim::world_origin_offset[0], EulerFluidSim::world_origin_offset[1], 0.0f};
    n_pos = DenormalizePosition(n_pos - halo) + static_cast<math::vec3<ValueType>>(offset);
    velocity = DenormalizePosition(velocity);

    CreateVelocityArrow(out,
                        arr_start_index,
                        ToWorldCoordinates(w, static_cast<math::vec3<float>>(n_pos)),
                        ToWorldCoordinates(w, static_cast<math::vec3<float>>(n_pos + velocity)),
                        head_size * velocity.magnitude());
}

void DrawFillInterpolatedVelocities(cudaGraphicsResource* velocities_resource,
                                    uint32_t interpolation_points,
                                    float head_size,
                                    cudaStream_t stream) {
    const dim3 grid{
        static_cast<uint32_t>(SimulationDomain::real_dimensions[0]) * interpolation_points / block_dimensions.x,
        static_cast<uint32_t>(SimulationDomain::real_dimensions[1]) * interpolation_points / block_dimensions.y,
        static_cast<uint32_t>(SimulationDomain::real_dimensions[2]) * interpolation_points / block_dimensions.z};

    cudaGraphicsMapResources(1, &velocities_resource, stream);
    math::vec3<float>* out = nullptr;
    size_t out_size = 0;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&out), &out_size, velocities_resource);
    assert(out_size >= EulerFluidSim::domains * Cardinality(SimulationDomain::real_dimensions) *
                           math::cube<size_t>(interpolation_points) * 6 * sizeof(math::vec3<float>) &&
           "velocity arrow buffer is too small for this sample count");

    for (size_t i = 0; i < EulerFluidSim::domains; ++i) {
        FillInterpolatedVelocities<<<grid, block_dimensions, 0, stream>>>(out, interpolation_points, i, head_size);
    }

    cudaGraphicsUnmapResources(1, &velocities_resource, stream);
    {
        auto err = cudaGetLastError();
        assert(err == cudaSuccess);
    }
}
