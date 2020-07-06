# Deep Learning C++ Libraries

- **07/01/2020 - Update Single Layer Perceptron**
- **07/03/2020 - Ongoing Variable, cudaMat, Functions**
- **07/05/2020 - Variable**

- **__ restrict __ in CUDA / Pointer Aliasing**
    - It is an optimizing way for pointer aliasing
    - C/C++ code cannot match FORTRAN performance, pointer aliasing is an important topic to understand when considering optimizations for C/C++ code.
    - Two pointers alias if the memory to which they point overlaps. When a compiler can't determine whether pointers alias, it has to assume that they do.
    - 


- **cuSPARSE**
    - CUDA sparse matrix library
    - contains a set of basic linear algebra subsolutions used for handling sparse matrices.
    - The library targets matrices with a number of zero elements which represent > 95% of the total entries.
    

- **cudaThreadSynchronize**
    - Wait for compute device to finish
    - Returns **cudaSuccess**
    - Blocks until the device has completed all preceding requested tasks.

- **CUBLASE Explanation**
    - CUBLAS? CUDA Basic Linear Algebra Subprogram
        - Exisiting BLAS is Fortran Library. CUDA uses the existing library and creates this cuBLAS.
        It is one of famous toolkits which NVIDIA provide as well as Fast Fourier Transform Library.
        In addition, it is free to use. 
    
    - Why CUBLAS?
        - Fast-accelerating Matrix operation is provided by cuBLAS. Exisiting openCL and Matrix example is processed by cuBLAS library.
    
    - What functions in CUBLAS?
        - Based on dimension, there are 3 levels functions.
        - Level 1: BLAS1 are functions that perform scalar, vector, and vector-vector operations.
        - Level 2: BLAS2 are functions that perform matrix-vector operations.
        - Level 3: BLAS3 are functions that perform matrix-matrix operations. 
    
    - Additional Information for CUBLAS
        - To use CUBLAS, call the functions with matrix or vector data GPU memory needs
        - For compatibility with Fortran, CUBLAS is column-major storage and 1-based indexing.
        - C and C++ are row-major storage and 0-base indexing. 
        - Therefore, for compatibility between CUBLAS and C&C++, indexing transformation macro is needed.
            
            -#define IDX2F(i, j, ld) (((j) - 1) * (ld)) + ((i)-1)
                
                - 1-based indexing
            
            -#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) 
                
                - 0-based indexing for C & C++ use
            
            - ld: matrix leading dimension         
    
    
    
    
- **CUDA Environement Comments**
    
    - Command environments
    
    nvcc -o main main.cpp -std=c++11 
    ./main
    
    -ldlib -L/usr/local/cuda-9.0/lib64 -lcudnn -lpthread -lcuda -lcudart -lcublas -lcurand -lcusolver -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -lopencv_dnn -lcudart -lgomp -lm -lstdc++ 

    - CMakeLists environments
    
    -#include_directories(/usr/local/cuda-9.0/targets/x86_64-linux/include)
    -#include_directories(/usr/local/cuda/targets/x86_64-linux/include)
    -#include_directories(/usr/local/cuda-9.0/lib64)
    -#include_directories(/usr/local/cuda/extras/CUPTI/lib64)
    -#include_directories(/usr/local/cuda-9.0)
    -#find_package(CUDA REQUIRED)
    -#include_directories("${CUDA_INCLUDE_DIRS}")
    
    
- **Reference**
https://github.com/takezo5096
