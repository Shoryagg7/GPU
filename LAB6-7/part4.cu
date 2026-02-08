// curand_demo.cu
#include <stdio.h>
#include <curand.h>
#include <cuda_runtime.h>

#define NUM_NUMBERS 1000000  // 1 million random numbers

int main() {
    float *d_numbers;
    cudaMalloc(&d_numbers, NUM_NUMBERS * sizeof(float));
    
    // Create cuRAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Set seed
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    // Time the generation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Generate uniform random numbers [0,1)
    curandGenerateUniform(gen, d_numbers, NUM_NUMBERS);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cuRAND (Default) Time: %.3f ms\n", ms);
    
    // ---- Compare different generators ----
    // 1. Pseudo-random (XORWOW - default)
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    
    // 2. Quasi-random (Sobol)
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32);
    
    // 3. Normal Distribution
    curandGenerateNormal(gen, d_numbers, NUM_NUMBERS, 0.0f, 1.0f);
    
    curandDestroyGenerator(gen);
    cudaFree(d_numbers);
    return 0;
}