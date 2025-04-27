#ifndef _ERROR_CHECKS_H_
#define _ERROR_CHECKS_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define checkCUBLAS(status)                                         \
    do                                                              \
    {                                                               \
        std::stringstream _err;                                     \
        if (status != CUBLAS_STATUS_SUCCESS)                        \
        {                                                           \
            _err << "cuBLAS error code: " << status                 \
                 << " in " << __FILE__ << " at line " << __LINE__;  \
             throw std::runtime_error(_err.str());                  \
        }                                                           \
    } while (0)

#define checkCUDNN(status)                                         \
    do                                                             \
    {                                                              \
        std::stringstream _err;                                    \
        if (status != CUDNN_STATUS_SUCCESS)                        \
        {                                                          \
            _err << "cuDNN error: " << cudnnGetErrorString(status) \
                 << " in " << __FILE__ << " at line " << __LINE__; \
            throw std::runtime_error(_err.str());                  \
        }                                                          \
    } while (0)


#define checkCUDA(status)                                          \
    do                                                             \
    {                                                              \
        std::stringstream _err;                                    \
        if (status != cudaSuccess)                                 \
        {                                                          \
            _err << "CUDA error: " << cudaGetErrorString(status)   \
                 << " in " << __FILE__ << " at line " << __LINE__; \
            throw std::runtime_error(_err.str());                  \
        }                                                          \
    } while (0)


#endif
