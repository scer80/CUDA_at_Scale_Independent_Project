#ifndef _MLP_H_
#define _MLP_H_


#include <memory>
#include <numeric>
#include <vector>
#include <cublas_v2.h>

#include "activation.hpp"
#include "error_checks.hpp"
#include "fused_softmax_nll_loss.cuh"
#include "linear.hpp"
#include "optimizer.cuh"
#include "tensor_map.hpp"

template <typename T>
struct MLP {

    vector<int> layer_sizes;
    vector<Linear<T>> layers;
    vector<Activation<T>> activations;
    SoftmaxNLLLoss<float> softmax_nll_loss;

    MLP(
        const vector<int>& prefix_sizes,
        const vector<int>& sizes
    ) :
        layer_sizes(sizes),
        layers(),
        activations(),
        softmax_nll_loss(
            [&]() {
                vector<int> combined_sizes = prefix_sizes;
                combined_sizes.push_back(sizes.back());
                return combined_sizes;
            }()
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

    ~MLP() = default;

    void forward(
        cublasHandle_t& cublasHandle,
        cudnnHandle_t& cudnnHandle,
        T* input,
        int* target_labels,
        bool compute_probs,
        bool compute_loss        
    ) {
        T* layer_input = input;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].forward(cublasHandle, layer_input);
            activations[i].forward(
                cudnnHandle,
                layers[i].tensor_map.tensor_descriptor["output"],
                layers[i].tensor_map.data["output"]
            );
            layer_input = activations[i].tensor_map.data["output"];
            // layer_input = layers[i].tensor_map.data["output"];
        }
        softmax_nll_loss.forward(
            activations.back().tensor_map.data["output"], target_labels, compute_probs, compute_loss
        );
        // softmax_nll_loss.forward(
        //     layers.back().tensor_map.data["output"], target_labels, compute_probs, compute_loss
        // );
        checkCUDA(cudaDeviceSynchronize());
    }

    void backward(
        cublasHandle_t& cublasHandle,
        cudnnHandle_t& cudnnHandle,
        T* input,
        int* target_labels
    ) {
        softmax_nll_loss.backward(softmax_nll_loss.tensor_map.data["probs"], target_labels);

        T* layer_d_output = softmax_nll_loss.tensor_map.data["d_logits"];

        for (int i = layers.size() - 1; i > static_cast<int>(-1); --i) {
            // std::cout << "MLP " << i << std::endl;
            activations[i].backward(
                cudnnHandle,
                layers[i].tensor_map.tensor_descriptor["output"],
                layers[i].tensor_map.data["output"],
                layer_d_output
            );
            
            T* layer_input_ptr = input;
            if (i > 0) {
                layer_input_ptr = activations[i - 1].tensor_map.data["output"];
            }
            
            layers[i].backward(
                cublasHandle,
                layer_input_ptr,
                activations[i].tensor_map.data["d_input"]
            );
            layer_d_output = layers[i].tensor_map.data["d_input"];
            // layers[i].backward(
            //     cublasHandle,
            //     layer_input_ptr,
            //     layer_d_output
            // );
        }
    }

    void update_weights(float learning_rate) {
        for (size_t i = 0; i < layers.size(); ++i) {
            // Update weights
            int nb_weights = size_from_dims(layers[i].tensor_map.dims["weight"]);
            int weights_per_thread = 1;
            int num_threads = 256;
            int weights_per_block = num_threads * weights_per_thread;
            int num_blocks = (nb_weights + weights_per_block - 1) / weights_per_block;
            weight_update_kernel<<<num_blocks, num_threads>>>(
                layers[i].tensor_map.data["weight"],
                layers[i].tensor_map.data["d_weight"],
                learning_rate,
                nb_weights,
                weights_per_thread
            );
        }
    }

    void init_weights() {
        for (size_t i = 0; i < layers.size(); ++i) {
            int nb_weights = size_from_dims(layers[i].tensor_map.dims["weight"]);
            float factor = sqrt(6 / layers[i].tensor_map.dims["weight"].front());  // Uniform He 
            float weights_init[nb_weights];
            // initialize weights
            for (int j = 0; j < nb_weights; ++j) {
                float random_value = static_cast<float>(rand()) / RAND_MAX;
                random_value -= 0.5f;
                // random_value *= factor;
                weights_init[j] = random_value;
            }
            
            checkCUDA(cudaMemcpy(
                layers[i].tensor_map.data["weight"],
                weights_init,
                nb_weights * sizeof(float),
                cudaMemcpyHostToDevice
            ));
            checkCUDA(cudaDeviceSynchronize());
        }
    }
};

#endif
