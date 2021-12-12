#include <stdio.h>
#include <stdlib.h>

void kernel_atax(float* A, float* x, float* y, float* tmp, size_t size) {
    // printf("atax size=%d\n", size);
    const size_t NX = size;
    const size_t NY = size;
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            tmp[i] += A[i * NY + j] * x[j];
        }
        for (size_t j = 0; j < NY; j++) {
            y[j] += A[i * NY + j] * tmp[i];
        }
    }
}
