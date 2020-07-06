- **CUDA Thread Indexing**
    - Host (CPU)
        - Kernel1, Kernel2, ...
        - Kernel is connecting with Device Grid
        - <<< (blocks per grid), (threads per blocks) >>>
        
    - Device (GPU)
        - Grid1, Grid2, ...
            - Block(0, 0), Block(1, 0), ...,   
            Block(0, 1), Block(1, 1), ...,   
            ...
                - Thread(0, 0, 0), Thread(1, 0, 0), ... ,  
                Thread(0, 1, 0), Thread(1, 1, 0), ... ,
               
    - Example: Thread Indexing
        - 1D grid of 1D blocks
            ```
            int getGlobalIdx_1D_1D(){
                return blockIdx.x *blockDim.x + threadIdx.x;
            }
            ```
        - 1D grid of 2D blocks
            ```
            int getGlobalIdx_1D_2D(){
                return blockIdx.x * blockDim.x * blockDim.y
                    + threadIdx.y * blockDim.x + threadIdx.x;
            }
            ```
        - 1D grid of 3D blocks
            ```
            int getGlobalIdx_1D_3D(){
                return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
                    + threadIdx.z * blockDim.y * blockDim.x
                    + threadIdx.y * blockDim.x + threadIdx.x;
            }
            ```
          
        - 2D grid of 1D blocks
            ```
            int getGlobalIdx_2D_1D(){
                int blockId = blockIdx.y * gridDim.x + blockIdx.x;
                int threadId = blockId * blockDim.x + threadIdx.x;
                return threadId;
            }
            ```
                  
        - 2D grid of 2D blocks
            ```
            int getGlobalIdx_2D_2D(){
                int blockId = blockIdx.x + blockIdx.y * gridDim.x;
                int threadId = blockId * (blockDim.x * blockDim.y)
                     + (threadIdx.y * blockDim.x) + threadIdx.x;
                return threadId;
            }          
            ```
                  
        - 2D grid of 3D blocks
            ```
            int getGlobalIdx_2D_3D(){
                int blockId = blockIdx.x + blockIdx.y * gridDim.x;
                int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
                return threadId;
            }          
            ```
                  
        - 3D grid of 1D blocks
            ```
            int getGlobalIdx_3D_1D(){
                int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
                int threadId = blockId * blockDim.x + threadIdx.x;
                return threadId;
            }
            ```
                  
        - 3D grid of 2D blocks
            ```
            int getGlobalIdx_3D_2D(){
                int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
                int threadId = blockId * (blockDim.x * blockDim.y)
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
                return threadId;
            }
            ```
                  
        - 3D grid of 3D blocks
            ```
            int getGlobalIdx_3D_3D(){
                int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
                int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
                return threadId;
            }
            ```

- **cublasSgeam**
    - It is matrix-matrix operation. 
    - cublasStatus_t cublasSgeam(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n,
                                const float           *alpha,
                                const float           *A, int lda,
                                const float           *beta,
                                const float           *B, int ldb,
                                float           *C, int ldc)
    

- **Kernel Function**
    - Should define kernel functions as __ global __
    - When call the kernel fucntions, <<< (blocks per grid), (threads per blocks) >>>
    - Block is a group of threads
    - Grid is a group of blocks

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
    
    
    
- **Reference**

    https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf