#include <stdio.h>
#include <stdlib.h>

void kernel_mvt(float* a, float* x1, float* x2, float* y1, float* y2, size_t size) {
    // printf("mvt size=%d\n", size);
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            x1[i] = x1[i] + a[i * N + j] * y1[j];
        }
    }
    for (size_t k = 0; k < N; k++) {
        for (size_t l = 0; l < N; l++) {
            x2[k] = x2[k] + a[k * N + l] * y2[l];
        }
    }
}
