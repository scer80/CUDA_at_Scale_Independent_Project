#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <memory>
#include <numeric>
#include <vector>
#include <cublas_v2.h>

#include "error_checks.hpp"
#include "tensor_map.hpp"

template <typename T>
struct Activation {

    TensorMap<T> tensor_map;
    const T alpha = 1.0f;
    const T beta = 0.0f;
    cudnnActivationDescriptor_t activation_descriptor;

    Activation(const vector<int>& ios) :
        tensor_map({
            {"output", ios},
            {"d_input", ios}
        }),
        activation_descriptor()
    {
        checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
        checkCUDNN(cudnnSetActivationDescriptor(
            activation_descriptor,
            CUDNN_ACTIVATION_ELU,
            CUDNN_NOT_PROPAGATE_NAN,
            0.0));
    }

    ~Activation() = default;

    void forward(
        cudnnHandle_t& cudnnHandle,
        cudnnTensorDescriptor_t& input_tensor_descriptor,
        T* input_ptr
    ) {
        checkCUDNN(
            cudnnActivationForward(
                cudnnHandle,
                activation_descriptor,
                &alpha,
                input_tensor_descriptor,
                input_ptr,
                &beta,
                tensor_map.tensor_descriptor["output"],
                tensor_map.data["output"]
            )
        );

        checkCUDA(cudaDeviceSynchronize()); 
    }

    void backward(
        cudnnHandle_t& cudnnHandle,
        cudnnTensorDescriptor_t& input_tensor_descriptor,
        T* input_ptr,
        T* d_output_ptr
    ) {
        checkCUDNN(
            cudnnActivationBackward(
                cudnnHandle,
                activation_descriptor,
                &alpha,
                tensor_map.tensor_descriptor["output"],
                tensor_map.data["output"],                
                tensor_map.tensor_descriptor["output"],
                d_output_ptr,
                input_tensor_descriptor,
                input_ptr,
                &beta,
                tensor_map.tensor_descriptor["d_input"],
                tensor_map.data["d_input"]
            )
        );
        checkCUDA(cudaDeviceSynchronize());
    }
};

#endif
