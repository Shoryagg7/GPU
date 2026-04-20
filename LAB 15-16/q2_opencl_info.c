// Q2: OpenCL device and platform info functions
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define CHECK_CL(call)                                                       \
    do {                                                                     \
        cl_int err = (call);                                                 \
        if (err != CL_SUCCESS) {                                             \
            fprintf(stderr, "OpenCL error %s:%d: %d\n", __FILE__, __LINE__,   \
                    err);                                                    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main(void) {
    cl_uint platform_count = 0;
    CHECK_CL(clGetPlatformIDs(0, NULL, &platform_count));
    
    if (platform_count == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return EXIT_FAILURE;
    }
    
    cl_platform_id *platforms =
        (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
    if (!platforms) {
        fprintf(stderr, "Allocation failed.\n");
        return EXIT_FAILURE;
    }
    
    CHECK_CL(clGetPlatformIDs(platform_count, platforms, NULL));
    
    printf("Number of platforms: %u\n", platform_count);
    
    for (cl_uint p = 0; p < platform_count; ++p) {
        char info[256];
        CHECK_CL(clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME,
                                    sizeof(info), info, NULL));
        printf("Platform %u: %s\n", p, info);
        
        cl_uint device_count = 0;
        CHECK_CL(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL,
                                 &device_count));
        printf("  Device count: %u\n", device_count);
        
        if (device_count > 0) {
            cl_device_id *devices =
                (cl_device_id *)malloc(sizeof(cl_device_id) * device_count);
            if (!devices) {
                fprintf(stderr, "Allocation failed.\n");
                free(platforms);
                return EXIT_FAILURE;
            }
            
            CHECK_CL(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                     device_count, devices, NULL));
            
            for (cl_uint d = 0; d < device_count; ++d) {
                char device_name[256];
                cl_ulong mem_size;
                cl_uint max_cu;
                
                CHECK_CL(clGetDeviceInfo(devices[d], CL_DEVICE_NAME,
                                         sizeof(device_name), device_name, NULL));
                CHECK_CL(clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE,
                                         sizeof(mem_size), &mem_size, NULL));
                CHECK_CL(clGetDeviceInfo(devices[d],
                                         CL_DEVICE_MAX_COMPUTE_UNITS,
                                         sizeof(max_cu), &max_cu, NULL));
                
                printf("    Device %u: %s\n", d, device_name);
                printf("      Global memory: %lu MB\n", mem_size / (1024 * 1024));
                printf("      Compute units: %u\n", max_cu);
            }
            
            free(devices);
        }
    }
    
    free(platforms);
    return 0;
}
