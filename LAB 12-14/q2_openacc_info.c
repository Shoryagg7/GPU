// Q2: OpenACC info functions
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

static double wall_time(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(void) {
    ensure_device();

#ifdef _OPENACC
    printf("OpenACC version: %d\n", _OPENACC);
#else
    printf("OpenACC version: unknown\n");
#endif

    acc_device_t dev = acc_get_device_type();
    int device_count = acc_get_num_devices(dev);
    int device_num = acc_get_device_num(dev);

    printf("Device type: %d\n", (int)dev);
    printf("Device count: %d\n", device_count);
    printf("Current device: %d\n", device_num);
    printf("Timer tick: %.9f sec\n", 1.0 / (double)CLOCKS_PER_SEC);

    double start = wall_time();
    #pragma acc parallel
    {
        // Empty kernel to confirm runtime availability
    }
    #pragma acc wait
    double end = wall_time();

    printf("Empty kernel time: %.6f ms\n", (end - start) * 1000.0);

    return 0;
}
