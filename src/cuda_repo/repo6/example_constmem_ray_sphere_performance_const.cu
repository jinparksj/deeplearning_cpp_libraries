#include <stdlib.h>
#include "cuda.h"
#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"

#define RAND_MAX 2147483647
#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 100
#define DIM 1024

// center of sphere: x, y, z
// sphere radius
// color: r, b, g
struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
    //threadIdx / blockIdx -> set pixel location
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM/2);
    float oy = (y - DIM/2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }

    ptr[offset * 4 + 0] = (int) (r * 255);
    ptr[offset * 4 + 1] = (int) (g * 255);
    ptr[offset * 4 + 2] = (int) (b * 255);
    ptr[offset * 4 + 3] = 255;

}

//Sphere *s;


void EXAMPLE_CONSTMEM_RAYSPHERE_PERFORMANCE_CONST_REPO6() {
    //capture start time
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    //Allocate GPU memory for output bitmap
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    //Allocate sphere image memory
    HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

    //Allocate temporary sphere memory and initialize it
    Sphere *temp_s = (Sphere *) malloc(sizeof(Sphere) * SPHERES);

    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    //Copy sphere memory from Host to GPU
    //remove memory
//    HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>> (dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;

    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);

}