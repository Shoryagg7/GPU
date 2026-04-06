// Q1: Shared memory tiled matrix multiplication using CURAND data
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

#define M 512
#define N 512
#define K 512
#define TILE 16

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CHECK_CURAND(call)                                                   \
    do {                                                                     \
        curandStatus_t status = (call);                                      \
        if (status != CURAND_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuRAND error %s:%d: %d\n", __FILE__, __LINE__,   \
                    status);                                                 \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void matmul_tiled(const float *A, const float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    #pragma unroll
    for (int t = 0; t < K / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main() {
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    cudaEvent_t rngStart, rngStop;
    CHECK_CUDA(cudaEventCreate(&rngStart));
    CHECK_CUDA(cudaEventCreate(&rngStop));
    CHECK_CUDA(cudaEventRecord(rngStart));
    CHECK_CURAND(curandGenerateUniform(gen, d_A, M * K));
    CHECK_CURAND(curandGenerateUniform(gen, d_B, K * N));
    CHECK_CUDA(cudaEventRecord(rngStop));
    CHECK_CUDA(cudaEventSynchronize(rngStop));

    float rngMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&rngMs, rngStart, rngStop));
    printf("cuRAND generation time: %.3f ms\n", rngMs);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, M / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, start, stop));
    printf("Tiled matmul kernel time: %.3f ms\n", kernelMs);

    float *h_C = static_cast<float *>(malloc(bytesC));
    if (!h_C) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    printf("C[0]=%.6f, C[last]=%.6f\n", h_C[0], h_C[M * N - 1]);

    free(h_C);
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUDA(cudaEventDestroy(rngStart));
    CHECK_CUDA(cudaEventDestroy(rngStop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
