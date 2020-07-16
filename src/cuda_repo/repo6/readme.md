#REPO6


- repo6
    - How to use constant memory by using CUDA C
    - Characteristics of constant memory
    - How to evaluate the performance of CUDA application by using CUDA event
    - CUDA Memory Hierarchy
        ![CUDA Memory](../../img/gpu_memory_hierarchy.png)
    
- **Bottle Neck** of hundreds of ALU in GPU is caused by GPU's memory bandwidth, not total workload of GPU
    - It is not enough fast to come Input data into GPU
    - It is not matter of the number of ALU
    - It is important to reduce the memory workload

- NVIDIA HW normally provides 64KB constant memory
    - If an user implements constant memory instead of global memory, it is possible to reduce the memory bandwidth

- Ray Tracing Example by Using Constant Memory
    - Ray tracing: Method to make an object in 3D project to 2D image
    - Raserization method
        - To locate virtual camera, one point is selected in scene
        - To visualize an image, the camera has light sensor
        - It should be figured out which light contacts the light sensor
        - Each pixel in a result image should bs same about ray color and intensity, which is contacted to spot sensor
        - image --- view ray ----> scent object
    - Simply, the example is,
        - each pixel emits one view ray
        - it tracks which a part in a sphere is contacted by the ray
        - it records depth of the point contacted by the ray
                

