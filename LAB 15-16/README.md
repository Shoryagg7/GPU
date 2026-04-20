# Assignment 15-16: OpenCL Environment and Matrix Operations

This lab covers OpenCL installation, platform/device discovery, and basic matrix operations with error handling and timing.

## Objectives

1. Learn OpenCL installation and computing environment setup.
2. Query and display OpenCL platform/device info.
3. Perform matrix add and multiply operations using OpenCL kernels.

## OpenCL Installation and Environment

### Windows with NVIDIA GPU

1. **Install NVIDIA CUDA Toolkit** (includes OpenCL libraries).
   - Download from https://developer.nvidia.com/cuda-downloads
   - Installation adds OpenCL headers and libraries.

2. **Install OpenCL headers** (if not included):
   - `cl.h` should be in `%CUDA_PATH%\include`
   - OpenCL libraries in `%CUDA_PATH%\lib\x64` (for x64 systems)

3. **Set environment variables**:
   - `CUDA_PATH` → typically set by installer
   - Add to compilation flags: `-I%CUDA_PATH%\include -L%CUDA_PATH%\lib\x64 -lOpenCL`

### Linux

```bash
sudo apt-get install opencl-headers ocl-icd-opencl-dev nvidia-opencl-dev  # Ubuntu/Debian
sudo dnf install opencl-headers ocl-icd-devel  # Fedora
```

On Linux, compile with: `gcc -o program program.c -lOpenCL`

## Files

| File | Purpose | Notes |
| --- | --- | --- |
| q2_opencl_info.c | Platform and device discovery | Prints device info and capabilities |
| q3_opencl_matrix_ops.c | Matrix add and multiply | Uses `matrix_data.bin` |

## Input Data Files

- `matrix_data.bin` is created automatically on first run if it does not exist.
- File contains random floating-point data for matrices A and B.

## Build and Run

### Windows (with NVIDIA CUDA)

```bash
gcc q2_opencl_info.c -o q2_opencl_info -I%CUDA_PATH%\include -L%CUDA_PATH%\lib\x64 -lOpenCL
gcc q3_opencl_matrix_ops.c -o q3_opencl_matrix_ops -I%CUDA_PATH%\include -L%CUDA_PATH%\lib\x64 -lOpenCL

q2_opencl_info
q3_opencl_matrix_ops
```

### Linux

```bash
gcc q2_opencl_info.c -o q2_opencl_info -lOpenCL
gcc q3_opencl_matrix_ops.c -o q3_opencl_matrix_ops -lOpenCL

./q2_opencl_info
./q3_opencl_matrix_ops
```

## Expected Output

**q2_opencl_info** prints:
- Number of available platforms
- Platform names
- Devices per platform (CPU, GPU, etc.)
- Device properties (memory, compute units)

**q3_opencl_matrix_ops** prints:
- Kernel execution time (ms)
- Sample output values from matrices
- Auto-generates `matrix_data.bin` if missing

## Notes

- OpenCL supports heterogeneous compute (CPU, GPU, FPGA, etc.).
- Ensure at least one OpenCL-capable device is available on your system.
- Matrix dimensions (M, N, K) can be adjusted in the source code.
