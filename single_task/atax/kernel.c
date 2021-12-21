#include <stdio.h>
#include <stdlib.h>

void P4atax(float* A, float* x, float* y, float* tmp, int* size) {
    // printf("atax *size=%d\n", *size);
    const int NX = *size;
    const int NY = *size;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            tmp[i] += A[i * NY + j] * x[j];
        }
        for (int j = 0; j < NY; j++) {
            y[j] += A[i * NY + j] * tmp[i];
        }
    }
}
