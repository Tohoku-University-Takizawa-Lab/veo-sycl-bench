#include <stdio.h>
#include <stdlib.h>

void P3mm3(float* A, float* B, float* C, float* D, float* E, float* F, float* G, int *size) {
    // printf("*size %d\n", *size);
    const int NI = *size;
    const int NJ = *size;
    const int NK = *size;
    const int NL = *size;
    const int NM = *size;
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            E[i * NJ + j] = 0;
            for (int k = 0; k < NK; ++k) {
                E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NL; j++) {
            F[i * NL + j] = 0;
            for (int k = 0; k < NM; ++k) {
                F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
            }
        }
    }
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NL; j++) {
            G[i * NL + j] = 0;
            for (int k = 0; k < NJ; ++k) {
                G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
            }
        }
    }
}
