#include <stdio.h>
#include <stdlib.h>

void kernel_mm2(float* A, float* B, float* C, float* D, float* E, size_t size) {
    // printf("mm2_cpu size=%d\n", size);
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    const size_t NL = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            for (size_t k = 0; k < NK; ++k) {
                C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            E[i * NL + j] = 0;
            for (size_t k = 0; k < NJ; ++k) {
                E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
            }
        }
    }
}
