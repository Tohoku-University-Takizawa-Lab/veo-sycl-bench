#include <stdio.h>
#include <stdlib.h>

void kernel_bicg(float* A, float* r, float* s, float* p, float* q, size_t size) {
    // printf("bicg size=%d\n", size);
    const size_t NX = size;
    const size_t NY = size;
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            s[j] += r[i] * A[i * NY + j];
            q[i] += A[i * NY + j] * p[j];
        }
    }
}
