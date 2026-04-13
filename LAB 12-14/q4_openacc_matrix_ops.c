// Q4: Basic matrix operations with OpenACC (file-backed input)
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 256
#define N 256
#define K 256

static void ensure_device(void) {
    acc_device_t dev = acc_device_default;
    int count = acc_get_num_devices(dev);
    if (count <= 0) {
        fprintf(stderr, "No OpenACC devices available.\n");
        exit(EXIT_FAILURE);
    }
    acc_set_device_type(dev);
    acc_init(dev);
}

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

static double wall_time(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(void) {
    ensure_device();
    srand(1234);

    float *A = (float *)malloc(sizeof(float) * M * K);
    float *B = (float *)malloc(sizeof(float) * K * N);
    float *Cadd = (float *)malloc(sizeof(float) * M * N);
    float *Cmul = (float *)malloc(sizeof(float) * M * N);

    if (!A || !B || !Cadd || !Cmul) {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    load_or_create("matrix_data.bin", A, B);

    double start, end;

    #pragma acc data copyin(A[0:M*K], B[0:K*N]) copyout(Cadd[0:M*N], Cmul[0:M*N])
    {
        start = wall_time();
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                Cadd[i * N + j] = A[i * K + j] + B[i * N + j];
            }
        }
        #pragma acc wait
        end = wall_time();
        printf("Matrix add time: %.3f ms\n", (end - start) * 1000.0);

        start = wall_time();
        #pragma acc kernels loop collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                #pragma acc loop seq
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                Cmul[i * N + j] = sum;
            }
        }
        #pragma acc wait
        end = wall_time();
        printf("Matrix multiply time: %.3f ms\n", (end - start) * 1000.0);
    }

    printf("Cadd[0]=%.6f, Cmul[last]=%.6f\n", Cadd[0], Cmul[M * N - 1]);

    free(A);
    free(B);
    free(Cadd);
    free(Cmul);

    return 0;
}
