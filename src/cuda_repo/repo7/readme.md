#REPO7

- Summary
    - Like constant memory, texture memory is another variety of read-only memory that can improve performance and reduce 
    memory traffic when reads have certain access patterns.
    - The performance characteristics of texture memory
    - How to use one-dimensional texture memory with CUDA C
    - How to use two-dimensional texture memory with CUDA C
    
- Like constant memory, texture memory is cached on chip, so in some situations it will provide higher effective bandwidth by reducing memory requests to off-chip DRAM
    - Specifically, texture caches are designed for graphics applications where memory access patters exhibit a great deal of spatial locality
    - This roughly implies that a thread is likely to read from an address near the address that nearby threads read

- Simulation Example
    - Heat Transfer Model
    - T_NEW = T_OLD + k * (T_TOP + T_BOTTOM + T_LEFT + T_RIGHT - 4 * T_OLD)
    -  
