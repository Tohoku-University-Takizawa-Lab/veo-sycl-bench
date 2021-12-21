#include <stdio.h>
#include <stdlib.h>

void P4syrk(float* A, float* C, int *size) {
    // printf("syrk *size=%d\n", *size);
    const int N = *size;
    const int M = *size;
    float alpha = 123;
    float beta = 14512;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * M + j] *= beta;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < M; k++) {
                C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
            }
        }
    }
}
