# Assignment 8-11: CUDA Kernels and Memory Optimizations

This lab implements eight CUDA programs covering tiled matrix multiplication variants, reduction, sorting, and 1D/2D convolution kernels with proper GPU error handling and CUDA event timing.

## Objectives

1. Shared-memory tiled matrix multiplication with CURAND input data.
2. Thread-coarsened tiled matrix multiplication.
3. Tiled matrix multiplication with boundary checks.
4. Parallel sum reduction with shared memory tiling.
5. Bubble sort using shared memory with CURAND input data.
6. Even-odd sort using shared memory with CURAND input data.
7. 1D convolution with boundary handling.
8. 2D convolution using constant memory mask.

## Files

| File                                 | Purpose                          | Notes                      |
| ------------------------------------ | -------------------------------- | -------------------------- |
| q1_shared_tiled_matmul_curand.cu     | Tiled matmul using shared memory | CURAND-generated data      |
| q2_thread_coarsening_tiled_matmul.cu | Thread-coarsened tiled matmul    | Coarsening factor = 2      |
| q3_tiled_matmul_boundary.cu          | Tiled matmul with bounds checks  | Handles non-multiple sizes |
| q4_reduction_shared_tiling.cu        | Parallel sum reduction           | Multi-pass reduction       |
| q5_bubble_sort_shared_curand.cu      | Bubble sort in shared memory     | CURAND-generated data      |
| q6_even_odd_sort_curand.cu           | Even-odd sort in shared memory   | CURAND-generated data      |
| q7_convolution_1d_boundary.cu        | 1D convolution                   | Boundary handling          |
| q8_convolution_2d_constant.cu        | 2D convolution                   | Constant memory mask       |

## Requirements

CUDA toolkit (nvcc), CUDA-capable NVIDIA GPU, and a compatible driver. Q1/Q5/Q6 require cuRAND (included with the CUDA toolkit).

## Build and Run

Compile each program with `nvcc` from this folder:

```bash
nvcc q1_shared_tiled_matmul_curand.cu -o q1_shared_tiled_matmul_curand -lcurand
nvcc q2_thread_coarsening_tiled_matmul.cu -o q2_thread_coarsening_tiled_matmul
nvcc q3_tiled_matmul_boundary.cu -o q3_tiled_matmul_boundary
nvcc q4_reduction_shared_tiling.cu -o q4_reduction_shared_tiling
nvcc q5_bubble_sort_shared_curand.cu -o q5_bubble_sort_shared_curand -lcurand
nvcc q6_even_odd_sort_curand.cu -o q6_even_odd_sort_curand -lcurand
nvcc q7_convolution_1d_boundary.cu -o q7_convolution_1d_boundary
nvcc q8_convolution_2d_constant.cu -o q8_convolution_2d_constant
```

Run the executables directly, for example:

```bash
.\q1_shared_tiled_matmul_curand
```

Each program prints CUDA event timing and a small output sample (or a sorted check for the sorting kernels).
