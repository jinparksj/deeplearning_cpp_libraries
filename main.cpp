#include <iostream>
//#include "example_slp.h"
//#include "src/Variable/Variable.h"
//#include "src/cuda_repo/repo5/example_thread_sum.cu"
//#include "src/cuda_repo/repo5/example_thread_wave.cu"
//#include "src/cuda_repo/repo5/example_thread_inner_product.cu"
#include "src/cuda_repo/repo5/example_thread_julia.cu"
using namespace std;


int main() {
//    EXAMPLE_THREAD_WAVE_REPO5();
//    ExampleSLP();
//    THREAD_EXAMPLE_REPO5();
//    THREAD_EXAMPLE_REPO5_1();
//    THREAD_EXAMPLE_INNER_PRODUCT_REPO5();
    EXAMPLE_THREAD_JULIA_REPO5();
    cout << "Test" << endl;
    return 0;
}