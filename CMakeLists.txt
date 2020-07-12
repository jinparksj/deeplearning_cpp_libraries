cmake_minimum_required(VERSION 3.14)
project(LIB_CPPDL)
#project(LIB_CPPDL CUDA)

#set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 11)
include_directories(/usr/local/cuda-9.0/targets/x86_64-linux/include)
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
include_directories(/usr/local/cuda-9.0/lib64)
include_directories(/usr/local/cuda/extras/CUPTI/lib64)
include_directories(/usr/local/cuda-9.0)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_executable(
        LIB_CPPDL main.cpp
        src/slp/single_perceptron.h
        src/slp/single_perceptron.cpp
)

#src/slp/single_perceptron.h
#src/slp/single_perceptron.cpp
#src/Variable/Variable.cpp
#src/Variable/Variable.h
#src/cudaMat/cudaMat.h
#src/cudaMat/cudaMatSparse.h

#set_target_properties(
#        LIB_CPPDL
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)