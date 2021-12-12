#include <stdio.h>
#include <stdlib.h>

#define ALPHA 32412
#define BETA 2123
void kernel_gemm(float* A, float* B, float* C, size_t size) {
    // printf("gemm size=%d\n", size);
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            C[i * NJ + j] *= BETA;
            for (size_t k = 0; k < NK; ++k) {
                C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
}
