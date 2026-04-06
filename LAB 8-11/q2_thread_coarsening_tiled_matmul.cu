// Q2: Thread-coarsened tiled matrix multiplication
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

#define M 512
#define N 512
#define K 512
#define TILE 16
#define COARSEN 2

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static void fill_random(float *data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

__global__ void matmul_coarsen(const float *A, const float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE * COARSEN];

    int row = blockIdx.y * TILE + threadIdx.y;
    int colBase = blockIdx.x * (TILE * COARSEN) + threadIdx.x;

    float acc[COARSEN] = {0.0f};

    #pragma unroll
    for (int t = 0; t < K / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        #pragma unroll
        for (int i = 0; i < COARSEN; ++i) {
            Bs[threadIdx.y][threadIdx.x + i * TILE] =
                B[(t * TILE + threadIdx.y) * N + colBase + i * TILE];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a = As[threadIdx.y][k];
            #pragma unroll
            for (int i = 0; i < COARSEN; ++i) {
                acc[i] += a * Bs[k][threadIdx.x + i * TILE];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < COARSEN; ++i) {
        C[row * N + colBase + i * TILE] = acc[i];
    }
}

int main() {
    srand(1234);

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float *h_A = static_cast<float *>(malloc(bytesA));
    float *h_B = static_cast<float *>(malloc(bytesB));
    float *h_C = static_cast<float *>(malloc(bytesC));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    fill_random(h_A, M * K);
    fill_random(h_B, K * N);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(N / (TILE * COARSEN), M / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_coarsen<<<grid, block>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, start, stop));
    printf("Coarsened matmul kernel time: %.3f ms\n", kernelMs);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    printf("C[0]=%.6f, C[last]=%.6f\n", h_C[0], h_C[M * N - 1]);

    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
