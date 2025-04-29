#ifndef _MLP_H_
#define _MLP_H_


#include <memory>
#include <numeric>
#include <vector>
#include <cublas_v2.h>

#include "activation.hpp"
#include "error_checks.hpp"
#include "linear.hpp"
#include "softmax.hpp"
#include "tensor_map.hpp"

template <typename T>
struct MLP {

    vector<int> layer_sizes;
    vector<Linear<T>> layers;
    vector<Activation<T>> activations;
    Softmax<T> softmax;
    Softmax<T> log_softmax;

    MLP(
        const vector<int>& prefix_sizes,
        const vector<int>& sizes
    ) :
        layer_sizes(sizes),
        layers(),
        activations(),
        softmax(
            {reduce(prefix_sizes.begin(), prefix_sizes.end(), 1, multiplies<int>()), sizes.back(), 1, 1},
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL
        ),
        log_softmax(
            {reduce(prefix_sizes.begin(), prefix_sizes.end(), 1, multiplies<int>()), sizes.back(), 1, 1},
            CUDNN_SOFTMAX_LOG,
            CUDNN_SOFTMAX_MODE_CHANNEL
        )
    {
        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            vector<int> layer_input_size = vector<int>(prefix_sizes);
            vector<int> layer_output_size = vector<int>(prefix_sizes);
            layer_input_size.emplace_back(sizes[i]);
            layer_output_size.emplace_back(sizes[i + 1]);

            layers.emplace_back(layer_input_size, layer_output_size);
            activations.emplace_back(layer_output_size);
        }
    }

    ~MLP()
    {}

    void forward(
        cublasHandle_t& cublasHandle,
        cudnnHandle_t& cudnnHandle,
        T* input_ptr
    ) {
        T* layer_input = input_ptr;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].forward(cublasHandle, layer_input);
            activations[i].forward(
                cudnnHandle,
                layers[i].tensor_map.tensor_descriptor["output"],
                layers[i].tensor_map.data["output"]
            );
            layer_input = activations[i].tensor_map.data["output"];
        }
        softmax.forward(cudnnHandle, activations.back().tensor_map.data["output"]);
        log_softmax.forward(cudnnHandle, activations.back().tensor_map.data["output"]);
    }

    void backward(
        cublasHandle_t& cublasHandle,
        cudnnHandle_t& cudnnHandle,
        T* input_ptr,
        T* d_output_ptr
    ) {
        softmax.backward(cudnnHandle, input_ptr, d_output_ptr);
        log_softmax.backward(cudnnHandle, input_ptr, d_output_ptr);

        T* layer_d_output_ptr = log_softmax.tensor_map.data["d_input"];

        for (size_t i = layers.size() - 1; i > static_cast<size_t>(-1); --i) {
            activations[i].backward(
                cudnnHandle,
                layers[i].tensor_map.tensor_descriptor["output"],
                layers[i].tensor_map.data["output"],
                layer_d_output_ptr
            );
            T* layer_input_ptr = input_ptr;
            if (i > 0) {
                activations[i - 1].tensor_map.data["output"];
            }

            layers[i].backward(
                cublasHandle,
                layer_input_ptr,
                activations[i].tensor_map.data["d_input"]
            );
        }
    }
};

#endif
