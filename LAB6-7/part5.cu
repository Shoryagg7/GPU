// thread_scaling.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void matrixAdd(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

void runWithThreads(float *d_A, float *d_B, float *d_C, int threadsPerBlock) {
    // 1D thread configuration for simplicity
    dim3 block(threadsPerBlock);
    dim3 grid((N * N + threadsPerBlock - 1) / threadsPerBlock);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixAdd<<<grid, block>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Threads/Block: %4d | Grid Size: %6d | Time: %.3f ms\n",
           threadsPerBlock,
           (N * N + threadsPerBlock - 1) / threadsPerBlock,
           ms);
}

int main() {
    int size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    printf("%-20s %-15s %-12s\n", "Threads/Block", "Grid Size", "Time (ms)");
    printf("------------------------------------------------\n");
    
    // Test with increasing thread counts
    int threadCounts[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 6; i++) {
        runWithThreads(d_A, d_B, d_C, threadCounts[i]);
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}