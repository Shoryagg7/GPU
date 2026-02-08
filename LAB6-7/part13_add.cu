// matrix_add.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024  // Matrix size N x N

// GPU Kernel for Matrix Addition
__global__ void matrixAdd(float *A, float *B, float *C, int n) {
    // Each thread handles one element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

// CPU version for comparison
void matrixAddCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n * n; i++)
        C[i] = A[i] + B[i];
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;        // Host (CPU) matrices
    float *d_A, *d_B, *d_C;        // Device (GPU) matrices
    
    // Allocate CPU memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize matrices with random values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // ---- CPU Execution Time ----
    clock_t cpu_start = clock();
    matrixAddCPU(h_A, h_B, h_C, N);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;
    printf("CPU Time: %.3f ms\n", cpu_time);
    
    // Allocate GPU memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data from CPU → GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Define Grid and Block dimensions
    dim3 blockDim(16, 16);          // 16x16 = 256 threads per block
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);
    
    // ---- GPU Execution Time ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Time: %.3f ms\n", gpu_time);
    
    // Copy result back GPU → CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}