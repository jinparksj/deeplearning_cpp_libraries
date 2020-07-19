#include <iostream>
//#include "example_slp.h"
//#include "src/Variable/Variable.h"
//#include "src/cuda_repo/repo5/example_thread_sum.cu"
//#include "src/cuda_repo/repo5/example_thread_wave.cu"
//#include "src/cuda_repo/repo5/example_thread_inner_product.cu"
//#include "src/cuda_repo/repo5/example_thread_julia.cu"
//#include "src/cuda_repo/repo6/example_constmem_ray_sphere_nonconst.cu"
#include "src/cuda_repo/repo6/example_constmem_ray_sphere_const.cu"
//#include "src/cuda_repo/repo6/example_constmem_ray_sphere_performance.cu"
//#include "src/cuda_repo/repo6/example_constmem_ray_sphere_performance_const.cu"
using namespace std;
__constant__ Sphere s[SPHERES];


int main() {
    EXAMPLE_CONSTMEM_RAYSPHERE_REPO6(s);
//    EXAMPLE_CONSTMEM_RAYSPHERE_PERFORMANCE_CONST_REPO6();
//    EXAMPLE_CONSTMEM_RAYSPHERE_PERFORMANCE_REPO6();
//    EXAMPLE_CONSTMEM_RAYSPHERE_REPO6();
//    EXAMPLE_THREAD_WAVE_REPO5();
//    ExampleSLP();
//    THREAD_EXAMPLE_REPO5();
//    THREAD_EXAMPLE_REPO5_1();
//    THREAD_EXAMPLE_INNER_PRODUCT_REPO5();
//    EXAMPLE_THREAD_JULIA_REPO5();
    cout << "Test" << endl;
    return 0;
}