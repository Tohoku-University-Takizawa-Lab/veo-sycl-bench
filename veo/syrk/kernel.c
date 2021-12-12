#include <stdio.h>
#include <stdlib.h>

void kernel_syrk(float* A, float* C, size_t size) {
    // printf("syrk size=%d\n", size);
    const size_t N = size;
    const size_t M = size;
    float alpha = 123;
    float beta = 14512;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * M + j] *= beta;
        }
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < M; k++) {
                C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
            }
        }
    }
}
