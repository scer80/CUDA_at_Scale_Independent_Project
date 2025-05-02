#ifndef _OPTIMIZER_CUH_
#define _OPTIMIZER_CUH_


template <typename DataType, typename IndexType>
__global__ void weight_update_kernel(
    DataType* weights,
    const DataType* weight_updates,
    const DataType learning_rate,
    IndexType nb_weights,
    IndexType weights_per_thread
) {
    IndexType thread_index = blockIdx.x * blockDim.x + threadIdx.x +
        (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x +
        (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
    
    for (IndexType index = thread_index; index < (thread_index + weights_per_thread); ++index) {
        if (index >= nb_weights) return;
        weights[index] -= learning_rate * weight_updates[index];
    }
}

#endif
