#ifndef _FUSED_SOFTMAX_NLL_LOSS_CUH_
#define _FUSED_SOFTMAX_NLL_LOSS_CUH_

#include "tensor_map.hpp"

template <typename DataType, typename IndexType>
__global__ void softmax_nll_loss_forward_kernel(
    const DataType* logits,
    const IndexType* target_labels,
    DataType* probs,
    DataType* log_probs,
    DataType* nll,
    DataType* nll_sum,
    IndexType nb_samples,
    IndexType nb_classes,
    bool compute_probs,
    bool compute_loss
) {
    IndexType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_samples) return;

    DataType max_logit = logits[index * nb_classes];
    for (IndexType class_index = 1; class_index < nb_classes; ++class_index) {
        if (logits[index * nb_classes + class_index] > max_logit) {
            max_logit = logits[index * nb_classes + class_index];
        }
    }

    DataType sum_exp = 0.0f;
    for (IndexType class_index = 0; class_index < nb_classes; ++class_index) {
        sum_exp += exp(logits[index * nb_classes + class_index] - max_logit);
    }
    
    for (IndexType class_index = 0; class_index < nb_classes; ++class_index) {
        log_probs[index * nb_classes + class_index] = logits[index * nb_classes + class_index] - max_logit - log(sum_exp);
    }

    if (compute_probs) {
        for (IndexType class_index = 0; class_index < nb_classes; ++class_index) {
            probs[index * nb_classes + class_index] = exp(log_probs[index * nb_classes + class_index]);
        }
    }
    
    if (compute_loss) {
        nll[index] = 0.0f;
        for (IndexType class_index = 0; class_index < nb_classes; ++class_index) {
            if (class_index == target_labels[index]) {
                nll[index] -= log_probs[index * nb_classes + class_index];
            }
            atomicAdd(nll_sum, nll[index]);
        }
    }
}


template <typename DataType, typename IndexType>
__global__ void softmax_nll_loss_backward_kernel(
    const DataType* probs,
    const IndexType* target_labels,
    DataType *d_logits,
    IndexType nb_samples,
    IndexType nb_classes
) {
    IndexType index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_samples) return;

    for (IndexType class_index = 0; class_index < nb_classes; ++class_index) {
        if (class_index == target_labels[index]) {
            d_logits[index * nb_classes + class_index] = probs[index * nb_classes + class_index] - 1.0f;
        } else {
            d_logits[index * nb_classes + class_index] = probs[index * nb_classes + class_index];
        }
    }
}

template <typename T>
struct SoftmaxNLLLoss {
    
    TensorMap<T> tensor_map;
    int nb_threads_per_block;

    SoftmaxNLLLoss(
        const vector<int>& ios,
        int nb_threads_per_block = 256
    ) :
        tensor_map({
            {"logits", ios},
            {"probs", ios},
            {"log_probs", ios},
            {"nll", {1, 1}},
            {"nll_sum", {1, 1}},
            {"d_logits", ios}
        })
    {
        this->nb_threads_per_block = nb_threads_per_block;
    }

    ~SoftmaxNLLLoss() = default;

    void forward(
        const T* logits,
        const int* target_labels,
        bool compute_probs,
        bool compute_loss
    ) { 
        int nb_samples = reduce(tensor_map.dims["logits"].begin(), tensor_map.dims["logits"].end() - 1, 1, std::multiplies<int>());
        int nb_classes = tensor_map.dims["logits"].back();

        int nb_blocks = (nb_samples + nb_threads_per_block - 1) / nb_threads_per_block;

        softmax_nll_loss_forward_kernel<T, int><<<nb_blocks, nb_threads_per_block>>>(
            logits,
            target_labels,
            tensor_map.data["probs"],
            tensor_map.data["log_probs"],
            tensor_map.data["nll"],
            tensor_map.data["nll_sum"],
            nb_samples,
            nb_classes,
            compute_probs,
            compute_loss
        );
    }

    void backward(
        const T* probs,
        const int* target_labels
    ) {
        int nb_samples = reduce(tensor_map.dims["logits"].begin(), tensor_map.dims["logits"].end() - 1, 1, std::multiplies<int>());
        int nb_classes = tensor_map.dims["logits"].back();

        int nb_blocks = (nb_samples + nb_threads_per_block - 1) / nb_threads_per_block;

        softmax_nll_loss_backward_kernel<T, int><<<nb_blocks, nb_threads_per_block>>>(
            probs,
            target_labels,
            tensor_map.data["d_logits"],
            nb_samples,
            nb_classes
        );
    }
};

#endif
