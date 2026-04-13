// Q5: Bubble sort, odd-even sort, and bitonic sort in OpenACC
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define N 1024

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

static void load_or_create(const char *path, float *data) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fill_random(data, N);
        f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "Failed to create input file: %s\n", path);
            exit(EXIT_FAILURE);
        }
        fwrite(data, sizeof(float), N, f);
        fclose(f);
        return;
    }
    size_t read = fread(data, sizeof(float), N, f);
    fclose(f);
    if (read != N) {
        fprintf(stderr, "Input file size mismatch: %s\n", path);
        exit(EXIT_FAILURE);
    }
}

static bool is_sorted(const float *data, int count) {
    for (int i = 1; i < count; ++i) {
        if (data[i] < data[i - 1]) {
            return false;
        }
    }
    return true;
}

static double wall_time(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void bubble_sort(float *data, int count, double *time_ms) {
    double start = wall_time();
    #pragma acc data copy(data[0:count])
    {
        #pragma acc parallel
        {
            #pragma acc loop seq
            for (int i = 0; i < count - 1; ++i) {
                #pragma acc loop seq
                for (int j = 0; j < count - 1 - i; ++j) {
                    if (data[j] > data[j + 1]) {
                        float tmp = data[j];
                        data[j] = data[j + 1];
                        data[j + 1] = tmp;
                    }
                }
            }
        }
        #pragma acc wait
    }
    double end = wall_time();
    *time_ms = (end - start) * 1000.0;
}

static void odd_even_sort(float *data, int count, double *time_ms) {
    double start = wall_time();
    #pragma acc data copy(data[0:count])
    {
        for (int phase = 0; phase < count; ++phase) {
            int start_idx = phase % 2;
            #pragma acc parallel loop
            for (int i = start_idx; i < count - 1; i += 2) {
                if (data[i] > data[i + 1]) {
                    float tmp = data[i];
                    data[i] = data[i + 1];
                    data[i + 1] = tmp;
                }
            }
        }
        #pragma acc wait
    }
    double end = wall_time();
    *time_ms = (end - start) * 1000.0;
}

static void bitonic_sort(float *data, int count, double *time_ms) {
    double start = wall_time();
    #pragma acc data copy(data[0:count])
    {
        for (int k = 2; k <= count; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                #pragma acc parallel loop
                for (int i = 0; i < count; ++i) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        bool ascending = ((i & k) == 0);
                        if ((ascending && data[i] > data[ixj]) ||
                            (!ascending && data[i] < data[ixj])) {
                            float tmp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = tmp;
                        }
                    }
                }
            }
        }
        #pragma acc wait
    }
    double end = wall_time();
    *time_ms = (end - start) * 1000.0;
}

int main(void) {
    ensure_device();
    srand(1234);

    float *base = (float *)malloc(sizeof(float) * N);
    float *bubble = (float *)malloc(sizeof(float) * N);
    float *oddeven = (float *)malloc(sizeof(float) * N);
    float *bitonic = (float *)malloc(sizeof(float) * N);

    if (!base || !bubble || !oddeven || !bitonic) {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    load_or_create("sort_data.bin", base);
    for (int i = 0; i < N; ++i) {
        bubble[i] = base[i];
        oddeven[i] = base[i];
        bitonic[i] = base[i];
    }

    double bubble_ms = 0.0;
    double odd_ms = 0.0;
    double bitonic_ms = 0.0;

    bubble_sort(bubble, N, &bubble_ms);
    odd_even_sort(oddeven, N, &odd_ms);
    bitonic_sort(bitonic, N, &bitonic_ms);

    printf("Bubble sort time: %.3f ms, sorted=%s\n",
           bubble_ms, is_sorted(bubble, N) ? "true" : "false");
    printf("Odd-even sort time: %.3f ms, sorted=%s\n",
           odd_ms, is_sorted(oddeven, N) ? "true" : "false");
    printf("Bitonic sort time: %.3f ms, sorted=%s\n",
           bitonic_ms, is_sorted(bitonic, N) ? "true" : "false");

    free(base);
    free(bubble);
    free(oddeven);
    free(bitonic);

    return 0;
}
