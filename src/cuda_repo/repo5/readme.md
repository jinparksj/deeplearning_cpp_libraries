#REPO5

- Summary
    - How to call CUDA C Thread 
    - How to communicate between threads
    - How to synchronize threads running in parallel

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
    - dim3  

