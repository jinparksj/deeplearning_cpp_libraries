#include <device_launch_parameters.h>
#include "../../common/book.h"

#define N 10
#define N1 (33 * 1024)

__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

void THREAD_EXAMPLE_REPO5() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    //Allocate GPU memory
    HANDLE_ERROR(cudaMalloc((void **) &dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_c, N * sizeof(int)));

    //Array 'a' and 'b' set in CPU
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    //Copy array 'a' and 'b' to GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    //Kernel Function Call
    add<<<1, N >>> (dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void add1(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N1) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void THREAD_EXAMPLE_REPO5_1() {
    int a[N1], b[N1], c[N1];
    int *dev_a, *dev_b, *dev_c;

    //Allocate GPU memory
    HANDLE_ERROR(cudaMalloc((void **) &dev_a, N1 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_b, N1 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_c, N1 * sizeof(int)));

    //Array 'a' and 'b' set in CPU
    for (int i = 0; i < N1; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    //Copy array 'a' and 'b' to GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N1 * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N1 * sizeof(int), cudaMemcpyHostToDevice));

    //Kernel Function Call
    add1<<<128, 128>>> (dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N1 * sizeof(int), cudaMemcpyDeviceToHost));
    bool success = true;

    for (int i = 0; i < N1; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf("Error: %d + %d != %d \n", a[i], b[i], c[i]);
            success = false;
        }

    }
    if (success) printf("We did it\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}