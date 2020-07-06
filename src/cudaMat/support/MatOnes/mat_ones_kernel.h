#ifndef LIB_CPPDL_MAT_ONES_KERNEL_H
#define LIB_CPPDL_MAT_ONES_KERNEL_H

#include <cuda_runtime.h>

__global__ void mat_ones_kernel(const float *__restrict__ src, float *__restrict__ dst,
        int m, int n);

#ifndef __cplusplus
extern "C" {
#endif
    void mat_ones_kernel_exec(const float *src, float *dst, int m, int n);

#ifdef __cplusplus
};
#endif

#endif //LIB_CPPDL_MAT_ONES_KERNEL_H
