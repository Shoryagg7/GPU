// Q7: 1D convolution with boundary handling
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

__global__ void conv1d(const float *input, float *output, const float *mask,
                       int n, int maskWidth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        int radius = maskWidth / 2;
        for (int k = 0; k < maskWidth; ++k) {
            int inIdx = idx + k - radius;
            if (inIdx >= 0 && inIdx < n) {
                sum += input[inIdx] * mask[k];
            }
        }
        output[idx] = sum;
    }
}

int main() {
    srand(1234);

    const int n = 1 << 16;
    const int maskWidth = 9;

    float h_mask[maskWidth] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               4.0f, 3.0f, 2.0f, 1.0f};

    size_t bytes = n * sizeof(float);
    float *h_input = static_cast<float *>(malloc(bytes));
    float *h_output = static_cast<float *>(malloc(bytes));
    if (!h_input || !h_output) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    fill_random(h_input, n);

    float *d_input = nullptr;
    float *d_output = nullptr;
    float *d_mask = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_mask, maskWidth * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask, maskWidth * sizeof(float),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    conv1d<<<blocks, threads>>>(d_input, d_output, d_mask, n, maskWidth);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, start, stop));
    printf("1D convolution kernel time: %.3f ms\n", kernelMs);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("Output[0]=%.6f, Output[last]=%.6f\n", h_output[0], h_output[n - 1]);

    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_mask));

    return 0;
}
