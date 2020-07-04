#ifndef LIB_CPPDL_CUDAMAT_H
#define LIB_CPPDL_CUDAMAT_H

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <sstream>
#include <map>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <cublas_v2.h>


using namespace std;

#define IDX2F(i, j, ld) ((((j)) * (ld)) + ((i)))

#define FatalError(s) {                                                 \
    std::stringstream _where, _message;                                 \
    _where << __FILE__ << ':' << __LINE__;                              \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;   \
    std::cerr << _message.str() << "\nAborting...\n"                    \
    cudaDeviceReset();                                                  \
    exit(EXIT_FAILURE);                                                 \
}

#define checkCUDNN(status) {                                            \
    std::stringstream _error;                                           \
    if (status != CUDNN_STATUS_STATUS) {                                \
        _error << "CUDNN failure\nError : " << cudnnGetErrorString(status);\
        FatalError(_error.str());                                       \
    }                                                                   \
}                                                                       \

#define checkCudaErrors(status) {                                       \
    std::stringstream _error;                                           \
    if (status != 0) {                                                  \
        _error << "CUDA failure\nError: " << cudaGetErrorString(status);\
        FatalError(_error.str());                                       \
    }                                                                   \
}                                                                       \

#define checkCublasErrors(status) {                                     \
    std::stringstream _error;                                           \
    if (status != 0) {                                                  \
        _error << "CUBLAS failure\nError: " << stauts;                  \
        FatalError(_error.str());                                       \
    }                                                                   \
}                                                                       \

class MallocCounter {
public:
    int num = 0;

    void up() {
        num++;
    }

    void down() {
        num--;
    }

    int get() {
        return num;
    }
};

extern MallocCounter mallocCounter;

class cudaMat {
private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version) {
        ar &mHostArray;
        ar &rows;
        ar &cols;
    }

public:
    float *mDevice = NULL;
    float *mHost = NULL;
    vector<float> mHostArray;
    int rows = 0;
    int cols = 0;

    cublasHandle_t cudaHandle;

    cudaMat() {
        rows = 0;
        cols = 0;
//        cublasCreate()
    }


};

#endif //LIB_CPPDL_CUDAMAT_H
