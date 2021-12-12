#include <stdio.h>
#include <stdlib.h>

void kernel_syr2k(float* A, float* B, float* C, size_t size) {
    // printf("syrk2 size=%d\n", size);
    const size_t N = size;
    const size_t M = size;
    float alpha = 1;
    float beta = 1;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * N + j] *= beta;
        }
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < M; k++) {
                C[i * N + j] += alpha * A[i * M + k] * B[j * M + k];
                C[i * N + j] += alpha * B[i * M + k] * A[j * M + k];
            }
        }
    }
}
