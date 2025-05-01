#ifndef _NLL_LOSS_CUH_
#define _NLL_LOSS_CUH_

#include <memory>
#include <numeric>
#include <vector>

#include "error_checks.hpp"
#include "tensor_map.hpp"

template <typename DataType, typename IndexType>
__global__ void nll_loss_forward_kernel(
    const DataType* log_softmax,
    const IndexType* target_labels,
    DataType* nll,
    DataType* nll_sum,
    IndexType nb_samples,
    IndexType nb_classes
) {
    // Kernel implementation for NLL loss
    // This is a placeholder and should be replaced with actual implementation
    IndexType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_samples) return;
    nll[index] = 0.0f;
    for (IndexType class_index = 0; class_index < nb_classes; ++class_index) {
        if (class_index == target_labels[index]) {
            nll[index] -= log_softmax[index * nb_classes + class_index];
        }
        atomicAdd(nll_sum, nll[index]);
    }
}

template <typename DataType, typename IndexType>
__global__ void nll_loss_backward_kernel(
    const DataType* log_softmax,
    const IndexType* target_labels,
    DataType* d_log_softmax,
    IndexType nb_samples,
    IndexType nb_classes
) {
    IndexType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_samples) return;

    for (IndexType class_index = 0; class_index < nb_classes; ++class_index) {
        if (class_index == target_labels[index]) {
            d_log_softmax[index * nb_classes + class_index] = -1.0f;
        } else {
            d_log_softmax[index * nb_classes + class_index] = 0.0f;
        }
    }
}

template <typename T>
struct NLLLoss {

    TensorMap<T> tensor_map;
    const T alpha = 1.0f;
    const T beta = 0.0f;
    int nb_threads_per_block;

    NLLLoss(
        const vector<int>& ios,
        int nb_threads_per_block = 256
    ) :
        tensor_map({
            {"nll", ios},
            {"nll_sum", {1, 1}},
            {"d_log_softmax", ios}
        })
    {
        this->nb_threads_per_block = nb_threads_per_block;
    }

    ~NLLLoss() = default;

    void forward(
        T* log_softmax,
        int* target_labels
    ) {
        // Get the number of samples and classes
        int nb_samples = reduce(tensor_map.dims["nll"].begin(), tensor_map.dims["nll"].end() - 1, 1, std::multiplies<int>());
        int nb_classes = tensor_map.dims["nll"].back();

        int nb_blocks = (nb_samples + nb_threads_per_block - 1) / nb_threads_per_block;
        // Launch the kernel for NLL loss forward
        nll_loss_forward_kernel<<<nb_blocks, nb_threads_per_block>>>(
            log_softmax,
            target_labels,
            tensor_map.data["nll"],
            tensor_map.data["nll_sum"],
            nb_samples,
            nb_classes
        );
        checkCUDA(cudaDeviceSynchronize());
    }

    void backward(
        T* log_softmax,
        int* target_labels
    ) {
        // Get the number of samples and classes
        int nb_samples = reduce(tensor_map.dims["nll"].begin(), tensor_map.dims["nll"].end() - 1, 1, std::multiplies<int>());
        int nb_classes = tensor_map.dims["nll"].back();

        int nb_blocks = (nb_samples + nb_threads_per_block - 1) / nb_threads_per_block;
        // Launch the kernel for NLL loss backward
        nll_loss_backward_kernel<<<nb_blocks, nb_threads_per_block>>>(
            log_softmax,
            target_labels,
            tensor_map.data["d_log_softmax"],
            nb_samples,
            nb_classes
        );
        checkCUDA(cudaDeviceSynchronize());
    }
};

#endif
