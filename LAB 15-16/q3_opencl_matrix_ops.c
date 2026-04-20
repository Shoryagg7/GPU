// Q3: OpenCL matrix operations (add and multiply)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define M 256
#define N 256
#define K 256

#define CHECK_CL(call)                                                       \
    do {                                                                     \
        cl_int err = (call);                                                 \
        if (err != CL_SUCCESS) {                                             \
            fprintf(stderr, "OpenCL error %s:%d: %d\n", __FILE__, __LINE__,   \
                    err);                                                    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

const char *kernel_source =
    "__kernel void matmul_add(__global float *A, __global float *B,\n"
    "                         __global float *Cadd, __global float *Cmul,\n"
    "                         int m, int n, int k) {\n"
    "    int i = get_global_id(0);\n"
    "    int j = get_global_id(1);\n"
    "    if (i < m && j < n) {\n"
    "        int idx = i * n + j;\n"
    "        Cadd[idx] = A[idx] + B[idx];\n"
    "        float sum = 0.0f;\n"
    "        for (int x = 0; x < k; ++x) {\n"
    "            sum += A[i * k + x] * B[x * n + j];\n"
    "        }\n"
    "        Cmul[idx] = sum;\n"
    "    }\n"
    "}\n";

static void fill_random(float *data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

static void load_or_create(const char *path, float *A, float *B) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fill_random(A, M * K);
        fill_random(B, K * N);
        f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "Failed to create input file: %s\n", path);
            exit(EXIT_FAILURE);
        }
        fwrite(A, sizeof(float), M * K, f);
        fwrite(B, sizeof(float), K * N, f);
        fclose(f);
        return;
    }
    size_t readA = fread(A, sizeof(float), M * K, f);
    size_t readB = fread(B, sizeof(float), K * N, f);
    fclose(f);
    if (readA != M * K || readB != K * N) {
        fprintf(stderr, "Input file size mismatch: %s\n", path);
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    srand(1234);

    float *h_A = (float *)malloc(sizeof(float) * M * K);
    float *h_B = (float *)malloc(sizeof(float) * K * N);
    float *h_Cadd = (float *)malloc(sizeof(float) * M * N);
    float *h_Cmul = (float *)malloc(sizeof(float) * M * N);

    if (!h_A || !h_B || !h_Cadd || !h_Cmul) {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    load_or_create("matrix_data.bin", h_A, h_B);

    cl_uint platform_count = 0;
    CHECK_CL(clGetPlatformIDs(0, NULL, &platform_count));
    if (platform_count == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return EXIT_FAILURE;
    }

    cl_platform_id *platforms =
        (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
    CHECK_CL(clGetPlatformIDs(platform_count, platforms, NULL));

    cl_device_id device = NULL;
    for (cl_uint p = 0; p < platform_count && device == NULL; ++p) {
        cl_uint device_count = 0;
        CHECK_CL(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL,
                                 &device_count));
        if (device_count > 0) {
            cl_device_id *devices =
                (cl_device_id *)malloc(sizeof(cl_device_id) * device_count);
            CHECK_CL(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                     device_count, devices, NULL));
            device = devices[0];
            free(devices);
        }
    }

    if (!device) {
        fprintf(stderr, "No OpenCL device found.\n");
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    if (!context) {
        fprintf(stderr, "Failed to create context.\n");
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_command_queue queue =
        clCreateCommandQueue(context, device, 0, NULL);
    if (!queue) {
        fprintf(stderr, "Failed to create command queue.\n");
        clReleaseContext(context);
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source,
                                                    NULL, NULL);
    if (!program) {
        fprintf(stderr, "Failed to create program.\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_int build_err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (build_err != CL_SUCCESS) {
        fprintf(stderr, "Program build failed.\n");
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                              &log_size);
        char *log = (char *)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log, NULL);
            fprintf(stderr, "Build log:\n%s\n", log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_kernel kernel = clCreateKernel(program, "matmul_add", NULL);
    if (!kernel) {
        fprintf(stderr, "Failed to create kernel.\n");
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * M * K, NULL, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * K * N, NULL, NULL);
    cl_mem d_Cadd = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * M * N, NULL, NULL);
    cl_mem d_Cmul = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * M * N, NULL, NULL);

    if (!d_A || !d_B || !d_Cadd || !d_Cmul) {
        fprintf(stderr, "Failed to create device buffers.\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(platforms);
        return EXIT_FAILURE;
    }

    CHECK_CL(clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0,
                                   sizeof(float) * M * K, h_A, 0, NULL, NULL));
    CHECK_CL(clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0,
                                   sizeof(float) * K * N, h_B, 0, NULL, NULL));

    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A));
    CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B));
    CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_Cadd));
    CHECK_CL(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_Cmul));
    CHECK_CL(clSetKernelArg(kernel, 4, sizeof(cl_int), &(int){M}));
    CHECK_CL(clSetKernelArg(kernel, 5, sizeof(cl_int), &(int){N}));
    CHECK_CL(clSetKernelArg(kernel, 6, sizeof(cl_int), &(int){K}));

    size_t global_work_size[2] = {M, N};
    size_t local_work_size[2] = {16, 16};

    cl_event event;
    double start = (double)clock() / (double)CLOCKS_PER_SEC;
    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                     local_work_size, 0, NULL, &event));
    CHECK_CL(clWaitForEvents(1, &event));
    double end = (double)clock() / (double)CLOCKS_PER_SEC;
    printf("Kernel execution time: %.3f ms\n", (end - start) * 1000.0);

    CHECK_CL(clEnqueueReadBuffer(queue, d_Cadd, CL_TRUE, 0,
                                  sizeof(float) * M * N, h_Cadd, 0, NULL,
                                  NULL));
    CHECK_CL(clEnqueueReadBuffer(queue, d_Cmul, CL_TRUE, 0,
                                  sizeof(float) * M * N, h_Cmul, 0, NULL,
                                  NULL));

    printf("Cadd[0]=%.6f, Cmul[last]=%.6f\n", h_Cadd[0], h_Cmul[M * N - 1]);

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_Cadd);
    clReleaseMemObject(d_Cmul);
    clReleaseEvent(event);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_A);
    free(h_B);
    free(h_Cadd);
    free(h_Cmul);
    free(platforms);

    return 0;
}
