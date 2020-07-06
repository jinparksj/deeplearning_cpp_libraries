#ifndef LIB_CPPDL_MAT_MUL_ELEMENTWISE_KERNEL_H
#define LIB_CPPDL_MAT_MUL_ELEMENTWISE_KERNEL_H
//#include "../../../../../../../../usr/local/cuda-9.0/targets/x86_64-linux/include/cuda_runtime.h"

#include <cuda_runtime.h>

__global__ void mat_mul_elementwise_kernel(const float *__restrict__ src1,
        const float *__restrict__ src2,
        float *__restrict__ dst, const int m, const int n);

#ifdef __cpluscplus
extern "C" {
#endif
    void mat_mul_elementwise_kernel_exec(const float *src1, const float *src2,
            float *dst, const int m, const int n);
#ifdef __cpluscplus
};
#endif

#endif //LIB_CPPDL_MAT_MUL_ELEMENTWISE_KERNEL_H
