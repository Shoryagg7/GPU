# Assignment 6-7: GPU Acceleration and CUDA Performance

This assignment demonstrates how GPU acceleration changes performance for common numerical workloads using standalone CUDA/C++ programs. It compares CPU and GPU timings for matrix operations, evaluates cuRAND generation time, and explores performance as thread counts increase.

## Objectives

1. Perform matrix addition and multiplication with CPU and GPU timing.
2. Compare random number generation performance using different cuRAND configurations.
3. Evaluate matrix-add performance as thread counts increase.

## Files

part13_add.cu: matrix addition on CPU and GPU with timing  
part3_mul.cu: matrix multiplication using cuBLAS with GPU timing  
part4.cu: cuRAND generation with multiple configurations  
part5.cu: matrix addition timing with increasing thread counts

## Requirements

CUDA toolkit (nvcc)  
CUDA-capable NVIDIA GPU with a compatible driver  
Standard C++ toolchain  
cuRAND and cuBLAS libraries (included with the CUDA toolkit)

## How to Run

1. Compile each .cu file with nvcc and run the resulting executable.
2. Ensure cuRAND is linked for part4.cu and cuBLAS is linked for part3_mul.cu.
3. If needed, adjust matrix size or thread configuration constants inside the source files.

## What to Expect

You will see:

1. CPU and GPU timing output for matrix addition.
2. GPU timing output for matrix multiplication (cuBLAS).
3. Timing output for cuRAND generation using different configurations.
4. A summary comparing kernel time as threads per block increase.

## Notes

Large matrix sizes increase memory use and can affect performance or runtime stability. GPU timings are measured with synchronization to ensure accurate results.
