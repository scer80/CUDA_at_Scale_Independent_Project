#ifndef __TENSOR_MAP_H_
#define __TENSOR_MAP_H_

#include <numeric>
#include <unordered_map>
#include <vector>

#include "error_checks.hpp"

using namespace std;


int size_from_dims(const vector<int>& dims) {
    return reduce(dims.begin(), dims.end(), 1, multiplies<int>());
}


vector<int> strides_from_dims(const vector<int>& dims) {
    auto result = vector<int>(dims.size());
    if (!dims.empty()) {
        result[dims.size() - 1] = 1;
        for (int i = dims.size() - 2; i >= 0; --i) {
            result[i] = dims[i + 1] * result[i + 1];
        }
    }
    return result;
}


template <typename T>
struct TensorMap {

    unordered_map<string, vector<int> > dims;
    unordered_map<string, T*> data;
    unordered_map<string, cudnnTensorDescriptor_t> tensor_descriptor;

    TensorMap(unordered_map<string, vector<int> > dims) :
        dims(dims),
        data(),
        tensor_descriptor()
    {
        for (const auto& [key, value] : dims) {
            auto size = size_from_dims(value);
            auto strides = strides_from_dims(value);

            data[key] = nullptr;
            checkCUDA(cudaMalloc(&data[key], size * sizeof(T)));
            
            tensor_descriptor[key] = cudnnTensorDescriptor_t();
            checkCUDNN(cudnnCreateTensorDescriptor(&tensor_descriptor[key]));
            checkCUDNN(cudnnSetTensorNdDescriptor(
                tensor_descriptor[key],
                CUDNN_DATA_FLOAT,
                value.size(),
                value.data(),
                strides.data()
            ));
        }
    }

    ~TensorMap() = default;

    void deallocate() {
        for (const auto& [key, value] : data) {
            checkCUDA(cudaFree(value));
            checkCUDNN(cudnnDestroyTensorDescriptor(tensor_descriptor[key]));
        }
    }

    void set_data(const string& name, T* host_data) {
        cudaMemcpy(data[name], host_data, size_from_dims(dims[name]) * sizeof(T), cudaMemcpyHostToDevice);
    }

    void get_data(const string& name, T* host_data) {
        cudaMemcpy(host_data, data[name], size_from_dims(dims[name]) * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
};

#endif
