# GPU Computing Lab Repository

This repository contains my lab work for GPU computing, OpenMP, OpenACC, and CUDA/OpenCL exercises. It is organized by lab ranges so each folder focuses on a specific topic or assignment set.

## Student Information

- Name: Shorya Gupta
- Roll No: 102316062

## Repository Structure

- [LAB1-2](LAB1-2) - OpenMP basics and parallel programming exercises.
- [LAB3-4](LAB3-4) - Additional C++ / parallel programming assignments.
- [LAB5](LAB5) - Notebook-based work and lab notes.
- [LAB6-7](LAB6-7) - CUDA programming fundamentals and arithmetic kernels.
- [LAB 8-11](LAB%208-11) - CUDA optimization labs covering tiled multiplication, reduction, sorting, and convolution.
- [LAB 12-14](LAB%2012-14) - OpenACC information, arithmetic, matrix operations, and sorting.
- [LAB 15-16](LAB%2015-16) - OpenCL information and matrix operations.

## What You Will Find

- Parallel programming examples using OpenMP.
- GPU kernel implementations in CUDA.
- OpenACC programs for acceleration and matrix work.
- OpenCL programs for GPU information and matrix operations.
- Lab-specific notes and README files inside each folder.

## How To Use

Open the folder for the lab you want to study, read its local README, and compile or run the corresponding source file from that directory.

### Common Build Commands

For CUDA programs:

```bash
nvcc source.cu -o output
```

For OpenMP C++ programs:

```bash
g++ -fopenmp source.cpp -o output
```

For OpenACC or OpenCL programs, use the compiler or SDK recommended by your lab instructions.

## Notes

- Output order in parallel programs may vary from run to run.
- Some CUDA programs may require `cuRAND` and a CUDA-capable NVIDIA GPU.
- Check the local README in each folder for exact build steps and expected output.
