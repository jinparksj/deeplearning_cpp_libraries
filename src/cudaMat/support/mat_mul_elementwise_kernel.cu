#include "mat_mul_elementwise_kernel_exec.h"
#define BLOCK_SIZE 32

__global__ void mat_mul_elementwise_kernel(const float *__restrict__ src1,
                                           const float *__restrict__ src2,
                                           float *__restrict__ dst, const int m, const int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        dst[row * n + col] = src1[row * n + col] * src2
    }
}
