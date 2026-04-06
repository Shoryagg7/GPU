// Q3: Tiled matrix multiplication with boundary checks
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define M 513
#define N 507
#define K 509
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

static void fill_random(float *data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

__global__ void matmul_tiled_boundary(const float *A, const float *B, float *C,
                                      int m, int n, int k) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int tiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        int tiledCol = t * TILE + threadIdx.x;
        int tiledRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < m && tiledCol < k) ? A[row * k + tiledCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (tiledRow < k && col < n) ? B[tiledRow * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
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
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_tiled_boundary<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, start, stop));
    printf("Boundary-checked matmul kernel time: %.3f ms\n", kernelMs);

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
