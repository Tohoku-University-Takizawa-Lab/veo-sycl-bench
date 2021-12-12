#include <stdio.h>
#include <stdlib.h>

void kernel_mm3(float* A, float* B, float* C, float* D, float* E, float* F, float* G, size_t size) {
    // printf("size %d\n", size);
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    const size_t NL = size;
    const size_t NM = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            E[i * NJ + j] = 0;
            for (size_t k = 0; k < NK; ++k) {
                E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            F[i * NL + j] = 0;
            for (size_t k = 0; k < NM; ++k) {
                F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
            }
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            G[i * NL + j] = 0;
            for (size_t k = 0; k < NJ; ++k) {
                G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
            }
        }
    }
}
