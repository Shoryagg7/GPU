// Q8: 2D convolution using constant memory mask
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define WIDTH 256
#define HEIGHT 256
#define MASK_WIDTH 3

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__constant__ float d_mask[MASK_WIDTH * MASK_WIDTH];

static void fill_random(float *data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

__global__ void conv2d(const float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int radius = MASK_WIDTH / 2;
        for (int j = -radius; j <= radius; ++j) {
            for (int i = -radius; i <= radius; ++i) {
                int xx = x + i;
                int yy = y + j;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    float val = input[yy * width + xx];
                    float coeff =
                        d_mask[(j + radius) * MASK_WIDTH + (i + radius)];
                    sum += val * coeff;
                }
            }
        }
        output[y * width + x] = sum;
    }
}

int main() {
    srand(1234);

    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    float *h_input = static_cast<float *>(malloc(bytes));
    float *h_output = static_cast<float *>(malloc(bytes));
    if (!h_input || !h_output) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    fill_random(h_input, WIDTH * HEIGHT);

    float h_mask[MASK_WIDTH * MASK_WIDTH] = {
        1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f
    };
    CHECK_CUDA(cudaMemcpyToSymbol(d_mask, h_mask, sizeof(h_mask)));

    float *d_input = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x,
              (HEIGHT + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    conv2d<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float kernelMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelMs, start, stop));
    printf("2D convolution kernel time: %.3f ms\n", kernelMs);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("Output[0]=%.6f, Output[last]=%.6f\n", h_output[0],
           h_output[WIDTH * HEIGHT - 1]);

    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
