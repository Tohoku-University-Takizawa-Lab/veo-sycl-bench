#include <stdio.h>
#include <stdlib.h>

#define ALPHA 32412
#define BETA 2123
void P4gemm(float* A, float* B, float* C, int *size) {
    // printf("gemm *size=%d\n", *size);   
    const int NI = *size;
    const int NJ = *size;
    const int NK = *size;
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            C[i * NJ + j] *= BETA;
            for (int k = 0; k < NK; ++k) {
                C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
}
