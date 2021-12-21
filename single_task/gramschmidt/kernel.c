#include <stdio.h>
#include <stdlib.h>

void P11gramschmidt(float* A, float* R, float* Q, int *size) {
    // printf("gramschmidt *size=%d\n", *size);
    const int M = *size;
    const int N = *size;
    for (int k = 0; k < N; k++) {
        float nrm = 0;
        for (int i = 0; i < M; i++) {
            nrm += A[i * N + k] * A[i * N + k];
        }
        R[k * N + k] = sqrt(nrm);
        for (int i = 0; i < M; i++) {
            Q[i * N + k] = A[i * N + k] / R[k * N + k];
        }
        for (int j = k + 1; j < N; j++) {
            R[k * N + j] = 0;
            for (int i = 0; i < M; i++) {
                R[k * N + j] += Q[i * N + k] * A[i * N + j];
            }
            for (int i = 0; i < M; i++) {
                A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
            }
        }
    }
}
