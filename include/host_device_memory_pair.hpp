#ifndef _HOST_DEVICE_MEMORY_PAIR_H_
#define _HOST_DEVICE_MEMORY_PAIR_H_


template <typename T>
struct HostDeviceMemoryPair {
    
    int64_t size;
    T* device_ptr;
    T* host_ptr;

    HostDeviceMemoryPair(int64_t size): size(size) {
        cudaMalloc(&device_ptr, size * sizeof(T));
        cudaMallocHost(&host_ptr, size * sizeof(T));
    }

    ~HostDeviceMemoryPair() {
        cudaFree(device_ptr);
        cudaFreeHost(host_ptr);
    }

    void copy_host_to_device() {
        cudaMemcpy(device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copy_device_to_host() {
        cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

#endif
