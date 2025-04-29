#ifndef _LINEAR_H_
#define _LINEAR_H_

#include <memory>
#include <numeric>
#include <vector>
#include <cublas_v2.h>

#include "error_checks.hpp"
#include "tensor_map.hpp"

template <typename T>
struct Linear {

    TensorMap<T> tensor_map;
    const T alpha = 1.0f;
    const T beta = 0.0f;
    int M, K, N;

    Linear(const vector<int>& inputs, const vector<int>& outputs) :
        tensor_map({
            {"weight", {inputs.back(), outputs.back()}},
            {"output", outputs},
            {"d_weight", {inputs.back(), outputs.back()}},
            {"d_input", inputs}
        })
    {
        M = reduce(inputs.begin(), inputs.end() - 1, 1, multiplies<int>());
        K = inputs.back();
        N = outputs.back();
    }

    ~Linear() = default;

    void forward(cublasHandle_t& cublasHandle, T* input_ptr) {
        checkCUBLAS(
            cublasSgemm(
                cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                tensor_map.data["weight"], N,
                input_ptr, K,
                &beta,
                tensor_map.data["output"], N
            )
        );
        checkCUDA(cudaDeviceSynchronize());
    }

    void backward(cublasHandle_t& cublasHandle, T* input_ptr, T* d_output_ptr) {
        checkCUBLAS(
            cublasSgemm(
                cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                K, M, N,
                &alpha,
                tensor_map.data["weight"], N,
                d_output_ptr, N,
                &beta,
                tensor_map.data["d_input"], K
            )
        );
        checkCUBLAS(
            cublasSgemm(
                cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                N, K, M,
                &alpha,
                d_output_ptr, N,
                input_ptr, K,
                &beta,
                tensor_map.data["d_weight"], N
            )
        );
        checkCUDA(cudaDeviceSynchronize());
    }
};

#endif
