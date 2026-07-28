#include "simulation_kernel.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// TODO: combine multiple domains into a single buffer for small simulations
// Arbitrary cut off here is 16MB (2^24 B) (~2 million elements, depending on the size)

// W is always 6, and it is a small enough constant to combine domains together.
// This will result in x6 buffers size, but x6 fewer dispatches

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

void CreateApplySparseMatrixGraph(
    cudaGraph_t* graph,
    EulerFluidSim::CudaDispatchBuffers* d_buffers,
    SimulationDomain* domains,
    size_t domain_index,
    const std::unique_ptr<SimulationDomain::GridCoefficientsValues> SimulationDomain::* h_coefficients_matrix,
    const EulerFluidSim::CudaDispatchBuffers::SB<CoefficientsValues<double>> EulerFluidSim::CudaDispatchBuffers::*
        d_coefficients_matrix,
    const std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* h_source,
    const EulerFluidSim::CudaDispatchBuffers::SB<double> EulerFluidSim::CudaDispatchBuffers::* d_source) {
    using ValueType = SimulationDomain::GridValues::value_type;
    static constexpr size_t elements = SimulationDomain::GridValues::cardinality;
    static constexpr dim3 block_dimensions{8, 8, 8};
    static constexpr dim3 grid_dimensions{SimulationDomain::real_dimensions[0] / block_dimensions.x,
                                          SimulationDomain::real_dimensions[1] / block_dimensions.y,
                                          SimulationDomain::real_dimensions[2] / block_dimensions.z};
    static constexpr size_t vec_size = elements * sizeof(ValueType);
    static constexpr size_t mat_size = elements * sizeof(CoefficientsValues<ValueType>);

    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    cudaMemcpyAsync((d_buffers->*d_coefficients_matrix).GetSegment(domain_index),
                    (domains[domain_index].*h_coefficients_matrix)->flat_data(),
                    mat_size,
                    cudaMemcpyHostToDevice,
                    capture_stream);
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

void CreateApplyAGraph(cudaGraph_t* graph,
                       EulerFluidSim::CudaDispatchBuffers* d_buffers,
                       SimulationDomain* domains,
                       size_t domain_index) {
    CreateApplySparseMatrixGraph(graph,
                                 d_buffers,
                                 domains,
                                 domain_index,
                                 &SimulationDomain::coefficients_,
                                 &EulerFluidSim::CudaDispatchBuffers::coefficients,
                                 &SimulationDomain::search_,
                                 &EulerFluidSim::CudaDispatchBuffers::search);
}

void CreateApplyMGraph(cudaGraph_t* graph,
                       EulerFluidSim::CudaDispatchBuffers* d_buffers,
                       SimulationDomain* domains,
                       size_t domain_index) {
    CreateApplySparseMatrixGraph(graph,
                                 d_buffers,
                                 domains,
                                 domain_index,
                                 &SimulationDomain::preconditioner_,
                                 &EulerFluidSim::CudaDispatchBuffers::preconditioner,
                                 &SimulationDomain::residual_,
                                 &EulerFluidSim::CudaDispatchBuffers::residual);
}

cudaGraph_t CreateApplyMatrixGraph(EulerFluidSim::CudaDispatchBuffers* d_buffers,
                                   SimulationDomain* domains,
                                   void (*create_func)(cudaGraph_t* graph,
                                                       EulerFluidSim::CudaDispatchBuffers* d_buffers,
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

cudaGraph_t CreateApplyAGraph(EulerFluidSim::CudaDispatchBuffers* d_buffers, SimulationDomain* domains) {
    return CreateApplyMatrixGraph(d_buffers, domains, CreateApplyAGraph);
}

cudaGraph_t CreateApplyMGraph(EulerFluidSim::CudaDispatchBuffers* d_buffers, SimulationDomain* domains) {
    return CreateApplyMatrixGraph(d_buffers, domains, CreateApplyMGraph);
}

/*
// It is easier to use a sencil for halo cells, then try to stride over them in dot product kernel
SimulationDomain::GridValues::value_type RealCellsDotProduct(
    EulerFluidSim::CudaDispatchBuffers* d_buffers,
    SimulationDomain* domains,
    const std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* lhs,
    const std::unique_ptr<SimulationDomain::GridValues> SimulationDomain::* rhs) {
    using ValueType = SimulationDomain::GridValues::value_type;
    static constexpr size_t elements = SimulationDomain::GridValues::cardinality;
    static constexpr size_t threads = 256;
    static constexpr size_t blocks = elements / threads + (elements % threads != 0);
    ValueType result{};

    for (size_t i = 0; i < EulerFluidSim::domains; ++i) {
        ValueType partial_result{};
        cudaMemcpy(d_buffers->grid_buffers[0], (domains[i].*lhs)->flat_data(), elements * sizeof(ValueType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_buffers->grid_buffers[1], real_grid_stencil.flat_data(), elements * sizeof(ValueType), cudaMemcpyHostToDevice);

        // Apply stencil to one of them before
        MultiplyInplace<ValueType><<<blocks, threads>>>(d_buffers->grid_buffers[0], d_buffers->grid_buffers[1], elements);
        cudaMemcpy(d_buffers->grid_buffers[1], (domains[i].*rhs)->flat_data(), elements * sizeof(ValueType), cudaMemcpyHostToDevice);

        // Dot product
        cublasDdot(d_buffers->handle, elements, d_buffers->grid_buffers[0], 1, d_buffers->grid_buffers[1], 1, &partial_result);
        result += partial_result;
    }

    return result;
}
*/
