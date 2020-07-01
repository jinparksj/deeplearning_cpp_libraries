cmake_minimum_required(VERSION 3.14)
project(LIB_CPPDL)

set(CMAKE_CXX_STANDARD 14)

add_executable(LIB_CPPDL main.cpp src/slp/single_perceptron.h src/slp/single_perceptron.cpp)