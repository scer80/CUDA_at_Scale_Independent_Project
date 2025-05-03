#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <unordered_map>

#include "error_checks.hpp"
#include <mnist_dataloader.hpp>
#include "error_checks.hpp"
#include "mlp.hpp"

using namespace std;


tuple<int, int, float> parseCommandLineArgs(int argc, char* argv[]) {
    vector<string> args(argv, argv + argc);

    int epochs = 1;
    int batch_size = 1;
    float learning_rate = 1e-3;

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--epochs") {
            epochs = stoi(args[i + 1]);
        }
        if (args[i] == "--batch_size") {
            batch_size = stoi(args[i + 1]);
        }
        if (args[i] == "--lr") {
            learning_rate = stof(args[i + 1]);
        }
    }
    return {epochs, batch_size, learning_rate};
}


void preprocess_mnist_sample(
    float* input,
    int* target_labels,    
    const ImageData& data,
    int index
) {
    // Copy the image data to the input array
    for (int i = 0; i < MNIST_IMAGE_SIZE; ++i) {
        float val = static_cast<float>(data.images[index][i]) / 255.0f;
        val = val * 2.0f - 1.0f; // Normalize to [-1, 1]
        input[i] = val;
    }

    // Copy the label to the target_labels array
    target_labels[0] = data.labels[index];
}


void evaluation(
    cudnnHandle_t cudnnHandle,
    cublasHandle_t cublasHandle,
    MLP<float>& mlp,
    ImageData data,
    unordered_map<string, float*>& input,
    unordered_map<string, int*>& target_labels,
    int batch_size,
    int epoch,
    int epochs,
    const string& set_name
) 
{
    vector<int> predicted_labels(data.images.size());
    float predicted_probs[MNIST_NB_CLASSES];
    int nb_batches = data.images.size() / batch_size;
    for (int batch_index = 0; batch_index < nb_batches; ++batch_index) {
        int start_index = batch_index * batch_size;
        int end_index = start_index + batch_size;
        for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
            preprocess_mnist_sample(
                input["host"] + sample_index * MNIST_IMAGE_SIZE,
                target_labels["host"] + sample_index,
                data,
                start_index + sample_index
            );
        }
        
        checkCUDA(cudaMemcpy(input["device"], input["host"], batch_size * MNIST_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        checkCUDA(cudaMemcpy(target_labels["device"], target_labels["host"], batch_size * sizeof(int), cudaMemcpyHostToDevice));
        
        mlp.forward(cublasHandle, cudnnHandle, input["device"], target_labels["device"], true, false);
        
        for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
            cudaMemcpy(
                predicted_probs,
                mlp.softmax_nll_loss.tensor_map.data["probs"] + sample_index * MNIST_NB_CLASSES,
                MNIST_NB_CLASSES * sizeof(float),
                cudaMemcpyDeviceToHost
            );
            int predicted_label = std::distance(
                predicted_probs, 
                std::max_element(predicted_probs, predicted_probs + MNIST_NB_CLASSES)
            );
            predicted_labels[start_index + sample_index] = predicted_label;
        }
        int correct_predictions = 0;
        for (int sample_index = 0; sample_index < end_index; ++sample_index) {
            if (predicted_labels[sample_index] == data.labels[sample_index]) {
                ++correct_predictions;
            }
        }
        float accuracy = static_cast<float>(correct_predictions) / end_index;

        std::cout << "\r\033[32mEpoch [" << epoch + 1 << "/" << epochs << "]\033[0m";
        std::cout << "\033[32m [" << end_index << "/" << nb_batches * batch_size << "] \033[0m";
        std::cout << "\033[33m " << set_name << " accuracy: "<< std::fixed << std::setprecision(2) << accuracy * 100.0f << "%\033[0m";
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    auto [epochs, batch_size, learning_rate] = parseCommandLineArgs(argc, argv);
    
    unsigned int seed = std::random_device{}();

    auto data = unordered_map<string, ImageData>();
    std::mt19937 rng(seed);

    std::cout << "\033[32mReading MNIST data...\033[0m" << std::flush;
    try {
        MnistDataloader loader(
            "data/MNIST/raw/train-images-idx3-ubyte",
            "data/MNIST/raw/train-labels-idx1-ubyte",
            "data/MNIST/raw/t10k-images-idx3-ubyte",
            "data/MNIST/raw/t10k-labels-idx1-ubyte"
        );

        data["test"] = loader.read_test_data();
        data["train"] = loader.read_training_data();
    } catch (const exception& e) {
        cerr << "\033[31m\nError: " << e.what() << "\033[0m\n";
        return 1;
    }
    std::cout << "\033[32mdone.\n\033[0m";

    cudnnHandle_t cudnnHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));

    cublasHandle_t cublasHandle = nullptr;
    checkCUBLAS(cublasCreate(&cublasHandle));

    unordered_map<string, float*> input = {
        {"host", nullptr},
        {"device", nullptr}
    };
    unordered_map<string, int*> target_labels = {
        {"host", nullptr},
        {"device", nullptr}
    };
    input["host"] = (float*)malloc(batch_size * MNIST_IMAGE_SIZE * sizeof(float));
    target_labels["host"] = (int*)malloc(batch_size * sizeof(int));
    checkCUDA(cudaMalloc(&input["device"], batch_size * MNIST_IMAGE_SIZE * sizeof(float)));
    checkCUDA(cudaMalloc(&target_labels["device"], batch_size * sizeof(int)));

    vector<int> hidden_layer_sizes = {32};
    vector<int> mlp_sizes = {MNIST_IMAGE_SIZE};
    mlp_sizes.insert(mlp_sizes.end(), hidden_layer_sizes.begin(), hidden_layer_sizes.end());
    mlp_sizes.push_back(MNIST_NB_CLASSES);

    MLP<float> mlp({batch_size, 1}, mlp_sizes);
    mlp.init_weights();

    int nb_batches;
    bool compute_probs = true;
    bool compute_loss = true;
    
    std::vector<size_t> train_indices(data["train"].images.size());
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::shuffle(train_indices.begin(), train_indices.end(), rng);

        nb_batches = data["train"].images.size() / batch_size;
        
        for (int batch_index = 0; batch_index < nb_batches; ++batch_index) {
            int start_index = batch_index * batch_size;
            int end_index = start_index + batch_size;
            for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
                preprocess_mnist_sample(
                    input["host"] + sample_index * MNIST_IMAGE_SIZE,
                    target_labels["host"] + sample_index,
                    data["train"],                    
                    train_indices[start_index + sample_index]
                );
            }
            
            checkCUDA(cudaMemcpy(input["device"], input["host"], batch_size * MNIST_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            checkCUDA(cudaMemcpy(target_labels["device"], target_labels["host"], batch_size * sizeof(int), cudaMemcpyHostToDevice));
            
            mlp.forward(cublasHandle, cudnnHandle, input["device"], target_labels["device"], compute_probs, compute_loss);
            mlp.backward(cublasHandle, cudnnHandle, input["device"], target_labels["device"]);
            mlp.update_weights(learning_rate);

            float loss;
            cudaMemcpy(&loss, mlp.softmax_nll_loss.tensor_map.data["nll_mean"], sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << "\r\033[32mEpoch [" << epoch + 1 << "/" << epochs << "]\033[0m";
            std::cout << "\033[32m [" << end_index << "/" << nb_batches * batch_size << "] \033[0m";
            std::cout << "\033[33m Loss: "<< std::scientific << loss << "\033[0m";
        }

        std::cout << std::endl;

        // Evaluation
        evaluation(cudnnHandle, cublasHandle, mlp, data["train"], input, target_labels, batch_size, epoch, epochs, "Train");
        evaluation(cudnnHandle, cublasHandle, mlp, data["test"], input, target_labels, batch_size, epoch, epochs, "Test");
    }

    free(input["host"]);
    free(target_labels["host"]);
    checkCUDA(cudaFree(input["device"]));
    checkCUDA(cudaFree(target_labels["device"]));

    return 0;
}
