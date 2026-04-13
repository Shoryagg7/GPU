# Assignment 12-14: OpenACC Environment and Kernels

This lab covers OpenACC environment setup, device info queries, basic arithmetic/matrix operations, and sorting algorithms with timing measurements.

## Objectives

1. Learn OpenACC installation and compute environment basics.
2. Display OpenACC runtime info functions.
3. Perform arithmetic operations using `parallel` and `kernels` constructs.
4. Perform matrix operations using `parallel` and `kernels` constructs with file-backed input.
5. Implement bubble sort, odd-even sort, and bitonic sort with OpenACC timing.

## OpenACC Installation and Environment

1. Install a compiler with OpenACC support (NVIDIA HPC SDK recommended).
2. Ensure compiler binaries are on your `PATH`.
3. Verify with a simple build: `nvc -acc -Minfo=accel q2_openacc_info.c -o q2_openacc_info`.
4. On Linux, ensure the NVIDIA driver is installed and the GPU is visible (e.g., `nvidia-smi`).

## Files

| File | Purpose | Notes |
| --- | --- | --- |
| q2_openacc_info.c | OpenACC info functions | Prints device info and timing tick |
| q3_openacc_arithmetic.c | Arithmetic ops | `parallel` for add/sub, `kernels` for mul/div |
| q4_openacc_matrix_ops.c | Matrix add/multiply | Uses `matrix_data.bin` |
| q5_openacc_sorts.c | Bubble, odd-even, bitonic sort | Uses `sort_data.bin` |

## Input Data Files

- `matrix_data.bin` and `sort_data.bin` are created automatically on the first run if they do not exist.
- Files are generated with random floating-point data.

## Build and Run

Using NVIDIA HPC SDK:

```bash
nvc -acc -Minfo=accel q2_openacc_info.c -o q2_openacc_info
nvc -acc -Minfo=accel q3_openacc_arithmetic.c -o q3_openacc_arithmetic
nvc -acc -Minfo=accel q4_openacc_matrix_ops.c -o q4_openacc_matrix_ops
nvc -acc -Minfo=accel q5_openacc_sorts.c -o q5_openacc_sorts
```

Using GCC (OpenACC enabled):

```bash
gcc -fopenacc q2_openacc_info.c -o q2_openacc_info
gcc -fopenacc q3_openacc_arithmetic.c -o q3_openacc_arithmetic
gcc -fopenacc q4_openacc_matrix_ops.c -o q4_openacc_matrix_ops
gcc -fopenacc q5_openacc_sorts.c -o q5_openacc_sorts
```

Run an executable, for example:

```bash
./q2_openacc_info
```
