// Q4: Parallel sum reduction using shared memory
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

__global__ void reduce_sum(const float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < static_cast<unsigned int>(n)) {
        sum += input[idx];
    }
    if (idx + blockDim.x < static_cast<unsigned int>(n)) {
        sum += input[idx + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    srand(1234);

    const int n = 1 << 20;
    const int threads = 256;

    size_t bytes = n * sizeof(float);
    float *h_data = static_cast<float *>(malloc(bytes));
    if (!h_data) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    fill_random(h_data, n);

    double cpu_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        cpu_sum += h_data[i];
    }

    float *d_in = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice));

    int max_blocks = (n + threads * 2 - 1) / (threads * 2);
    float *d_buf1 = nullptr;
    float *d_buf2 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf1, max_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_buf2, max_blocks * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    int cur_n = n;
    const float *d_src = d_in;
    float *d_dst = d_buf1;

    while (true) {
        int blocks = (cur_n + threads * 2 - 1) / (threads * 2);
        reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_src, d_dst, cur_n);
        CHECK_CUDA(cudaGetLastError());
        if (blocks == 1) {
            break;
        }
        cur_n = blocks;
        d_src = d_dst;
        d_dst = (d_dst == d_buf1) ? d_buf2 : d_buf1;
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, start, stop));

    float gpu_sum = 0.0f;
    CHECK_CUDA(cudaMemcpy(&gpu_sum, d_dst, sizeof(float), cudaMemcpyDeviceToHost));

    printf("GPU sum: %.6f\n", gpu_sum);
    printf("CPU sum: %.6f\n", static_cast<float>(cpu_sum));
    printf("Reduction kernel time: %.3f ms\n", kernelMs);

    free(h_data);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_buf1));
    CHECK_CUDA(cudaFree(d_buf2));

    return 0;
}
