// Q5: Bubble sort using shared memory, data generated with CURAND
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

#define N 1024

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

__global__ void bubble_sort(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < n) {
        sdata[tid] = data[tid];
    }
    __syncthreads();

    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - 1 - i; ++j) {
            if (tid == j) {
                float a = sdata[j];
                float b = sdata[j + 1];
                if (a > b) {
                    sdata[j] = b;
                    sdata[j + 1] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < n) {
        data[tid] = sdata[tid];
    }
}

int main() {
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 5678ULL));
    CHECK_CURAND(curandGenerateUniform(gen, d_data, N));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    bubble_sort<<<1, N, N * sizeof(float)>>>(d_data, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, start, stop));
    printf("Bubble sort kernel time: %.3f ms\n", kernelMs);

    float *h_data = static_cast<float *>(malloc(N * sizeof(float)));
    if (!h_data) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool sorted = true;
    for (int i = 1; i < N; ++i) {
        if (h_data[i] < h_data[i - 1]) {
            sorted = false;
            break;
        }
    }
    printf("Sorted: %s\n", sorted ? "true" : "false");
    printf("First element: %.6f, last element: %.6f\n", h_data[0], h_data[N - 1]);

    free(h_data);
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
