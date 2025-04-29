#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include <memory>
#include <numeric>
#include <vector>
#include <cublas_v2.h>

#include "error_checks.hpp"
#include "tensor_map.hpp"

template <typename T>
struct Softmax {

    TensorMap<T> tensor_map;
    const T alpha = 1.0f;
    const T beta = 0.0f;
    cudnnSoftmaxAlgorithm_t algorithm;
    cudnnSoftmaxMode_t mode;

    Softmax(
        const vector<int>& ios,
        cudnnSoftmaxAlgorithm_t _algorithm,
        cudnnSoftmaxMode_t _mode
    ) :
        tensor_map({
            {"output", ios},
            {"d_input", ios}
        })
    {
        algorithm = _algorithm;
        mode = _mode;        
    }

    ~Softmax() = default;

    void forward(
        cudnnHandle_t& cudnnHandle,
        T* input_ptr
    ) {
        checkCUDNN(
            cudnnSoftmaxForward(
                cudnnHandle,
                algorithm,
                mode,
                &alpha,
                tensor_map.tensor_descriptor["output"],
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
        T* input_ptr,
        T* d_output_ptr
    ) {
        checkCUDNN(
            cudnnSoftmaxBackward(
                cudnnHandle,
                algorithm,
                mode,
                &alpha,
                tensor_map.tensor_descriptor["output"],
                tensor_map.data["output"],                
                tensor_map.tensor_descriptor["output"],
                d_output_ptr,
                &beta,
                tensor_map.tensor_descriptor["d_input"],
                tensor_map.data["d_input"]
            )
        );
        checkCUDA(cudaDeviceSynchronize());
    }
};

#endif
