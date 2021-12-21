#include <stdio.h>
#include <stdlib.h>

void P5syrk2(float* A, float* B, float* C, int *size) {
    // printf("syrk2 *size=%d\n", *size);
    const int N = *size;
    const int M = *size;
    float alpha = 1;
    float beta = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] *= beta;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < M; k++) {
                C[i * N + j] += alpha * A[i * M + k] * B[j * M + k];
                C[i * N + j] += alpha * B[i * M + k] * A[j * M + k];
            }
        }
    }
}
