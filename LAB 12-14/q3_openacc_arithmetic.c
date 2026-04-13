// Q3: Basic arithmetic operations using OpenACC parallel and kernels constructs
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 20)

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

static void fill_random(float *data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

static double wall_time(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(void) {
    ensure_device();
    srand(1234);

    float *a = (float *)malloc(sizeof(float) * N);
    float *b = (float *)malloc(sizeof(float) * N);
    float *add = (float *)malloc(sizeof(float) * N);
    float *sub = (float *)malloc(sizeof(float) * N);
    float *mul = (float *)malloc(sizeof(float) * N);
    float *divv = (float *)malloc(sizeof(float) * N);

    if (!a || !b || !add || !sub || !mul || !divv) {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    fill_random(a, N);
    fill_random(b, N);
    for (int i = 0; i < N; ++i) {
        b[i] += 0.1f;
    }

    double start, end;

    #pragma acc data copyin(a[0:N], b[0:N]) copyout(add[0:N], sub[0:N], mul[0:N], divv[0:N])
    {
        start = wall_time();
        #pragma acc parallel loop
        for (int i = 0; i < N; ++i) {
            add[i] = a[i] + b[i];
        }
        #pragma acc parallel loop
        for (int i = 0; i < N; ++i) {
            sub[i] = a[i] - b[i];
        }
        #pragma acc wait
        end = wall_time();
        printf("Parallel add/sub time: %.3f ms\n", (end - start) * 1000.0);

        start = wall_time();
        #pragma acc kernels loop
        for (int i = 0; i < N; ++i) {
            mul[i] = a[i] * b[i];
        }
        #pragma acc kernels loop
        for (int i = 0; i < N; ++i) {
            divv[i] = a[i] / b[i];
        }
        #pragma acc wait
        end = wall_time();
        printf("Kernels mul/div time: %.3f ms\n", (end - start) * 1000.0);
    }

    printf("add[0]=%.6f, div[last]=%.6f\n", add[0], divv[N - 1]);

    free(a);
    free(b);
    free(add);
    free(sub);
    free(mul);
    free(divv);

    return 0;
}
