#CUDA Programming Personal Repositories



- repo5
    - How to call CUDA C Thread 
    - How to communicate between threads
    - How to synchronize threads running in parallel
    - 
    ![repo5_example_thread_wave](../img/repo5_thread_wave.png)
    
    - __ syncthreads() performance
        - No Synchronize Threads
            ![No syncthreads](../img/repo5_no_syncthreads.png)
           
        - Synchronize Threads
            ![syncthreads](../img/repo5_syncthreads.png)    




- Compile Command
    - nvcc -o main main.cpp -lGL -lglut -x cu