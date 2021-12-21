#include <stdio.h>
#include <stdlib.h>

void P3mm2(float* A, float* B, float* C, float* D, float* E, int* size) {
    // printf("mm2 *size=%d\n", *size);
    int NI = *size;
    int NJ = *size;
    int NK = *size;
    int NL = *size;
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            for (int k = 0; k < NK; ++k) {
                C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NL; j++) {
            E[i * NL + j] = 0;
            for (int k = 0; k < NJ; ++k) {
                E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
            }
        }
    }
}
