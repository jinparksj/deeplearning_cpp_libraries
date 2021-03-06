#REPO5

- Summary
    - How to call CUDA C Thread 
    - How to communicate between threads
    - How to synchronize threads running in parallel
    - Wave Julia Set
    ![repo5_example_thread_wave](../../img/repo5_thread_wave.png)
    
    - __ syncthreads() performance
        - No Synchronize Threads
            ![No syncthreads](../../img/repo5_no_syncthreads.png)
           
        - Synchronize Threads
            ![syncthreads](../../img/repo5_syncthreads.png)    

- The number of Kernel copies is limited up to 65563 in my setting of CUDA and GPU.

- Parallel copies are called as 'blocks'
    - CUDA runtime allows blocks to divide as 'threads'
    - kernel functions, <<< (blocks per grid), (threads per blocks) >>>
    - The first parameter is changed to the number of blocks I would like to run in parallel\
    - It can run blocks in parallel
    - add <<<N, 1>>> (dev_a, dev_b, dev_c); : Run blocks of Vector N 
        - N blocks * 1 thread / blocks = N parallel threads
        - So, it can run two threads with N / 2 blocks or four threads with N / 4 blocks
    - add <<<1, N>>> (dev_a, dev_b, dev_c);
        - One block and N threads

- __ global __ : A function is running in a device, GPU, not CPU
        
- blockIdx
    - It is not declared before. But, it is declared in CUDA run-time built-in variables.
    - It is used for 'block index value', running device code (GPU) in real-time
    - Why not blockIdx, but blockIdx.x?
        - CUDA C defines 2D blocks as one group
        - It is convenient to use Matrix operation and image processing
        - If the number of blocks is N,
            - blockIdx.x is 0 to N - 1
            - int tid = blockIdx.x; // tid is 0 ~ N - 1
            - Block's each dimension should not be over 65,535 by hardware limitation
            
- dim3
    - dim3 grid(DIM, DIM)
    - dim3 is not standard C type
    - It is defined for convenient usage of capsulizing tuple of multidimensional elements by CUDA run-time headers
    
- The number of limited threads
    - Device structure - maxThreadsPerBlock member - shows the maximum value of number of threads
    - Normally, the max value is 512
    - How can it execute vector sum, if the size of vector is more than 512?
        - At kernel, index calculation should be adjusted
        - int tid = threadIdx.x + blockIdx.x * blockDim.x;

- blockDim
    - It is fixed value for all blocks
    - It contains the number of threads for each block dimension 
    - If we want to use only one dimension, we can use blockDim.x
    - It can be similar with gridDim. But, gridDim is the number of each dimension blocks for whole grids
    - gridDim is 2D and blockDim is 3D
        - CUDA run-time can run one 2D grid for 3D thread array blocks
        
- 128 Threads per Block, 128 Threads Kernel Function

    - ```
        add<<< (N+127) / 128, 128 >>> (dev_a, dev_b, dev_c);
        if (tid < N)
            c[tid] = a[tid] + b[tid];
        ```

- The number of blocks is limited by hardware
    - The number of Grid for Blocks should be less than 65,535
    - If N/128 blocks run for vector sum, the number of vector is 65,535 * 128 = 8,388,480
        - It will fail
        - Solution: Each thread is done with one task and then, it should increase the index up to total number of threads in a grid.
            - blockDim.x * gridDim.x 
            - ```
              __global__ void add(int *a, int *b, int *c) {
                    int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    while (tid < N) {
                        c[tid] = a[tid] + b[tid];
                        tid += blockDim.x * gridDim.x;     
                    }
              }
              ``` 
- GPU tasks should consider the number of threads based on the processor core numbers
    - Decoupling the parallelization

- Thread Indexing
    - 2D Threads / 2D Blocks
    - ```
      int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
      int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
      int tid_offset = x + y * blockDim.x * gridDim.x;
      ``` 
    - tid_offset: output buffer offset -> it should be linearized like vector operation        
        
- Reasons why blocks divide to several threads?
    - CUDA C Shared Memory
    - __ shared __
    - CUDA C compiler generates copies of variables for each blocks running in GPU
    - Threads in each block can share memory. But, they cannot modify copies of variables in other blocks.
    - Shared memory provides a way of communicating and cooperating between threads in blocks
    - Shared memory buffer is in GPU physically
    - Communication and synchronization between threads are important due to race condition
    
- Shared Memory Buffer
    - __ shared __ floar cache[threadsPerBlock];
        - Shared memory buffer: This buffer is used for results of operating by threads
        - It looks like 'static' or 'volatile' at standard C
    - The offset of shared memory cache is just thread index
        - int chacheIndex = threadIdx.x;
        - The shared memory offset doesn't need to include blockIdx
        - Because each block has a copy of this shared memory
    - If input vector size is not multiple of threads in the block, there is a possibility not to use some of shared memory

- __ syncthreads() : Before reading values from shared memory buffer, it should be checked whether the task that shared memory is written is done.   
    syncthreads() guarantees the task is done.

- Reduction: Parallel computing
    - time complexity is reduced from O(N) to O(logN) due to parallel computing

- Why Device(GPU) returns control authority to Host (CPU), even though inner product - reduction operation - is not finished in GPU?
    - Inner product is reduction operation
    - The number of input is larger than the number of output -> The characteristics of 'reduction'
    - Inner product always returns one output no matter how many inputs are
    - GPU, like big parallel operating device, spends too much resources at final stage of 'reduction'
    - Because the size of data set is way smaller than the beginning of operation
    - So, it is better for Host - CPU - to finalize the operation to save resources in GPU
    - Instead of finalizing the reduction operation, GPU can spend other tasks efficiently

- REMEMBER in this repo
    - Allocate host and device memory
    - By using cudaMemcpy(), copy a[] and b[] - CPU host array - to GPU device memory
    - Predefine threads per block and blocks per grid
    - Decide blocks per grid: min(32, (N + threadsPerBlocks - 1) / threadsPerBlocks) //N: the number of data elements

- __ syncthread()
    - CUDA Architecture guarantees,
        - Hold any thread tasks by __ syncthread(), if there are any threads, still running in GPU
        - Some threads should wait for that other threads are done
    - Before running next instruction, it guarantees that data is ready 
    
    
             
        
- Compile Command
    - **nvcc -o main main.cpp -lGL -lglut -x cu**
    - **./main**
    