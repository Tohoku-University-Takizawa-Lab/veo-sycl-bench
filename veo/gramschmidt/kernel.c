#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void gramschmidt(float* A, float* R, float* Q, size_t size) {
    // printf("gramschmidt size=%d\n", size);
    const size_t M = size;
    const size_t N = size;
    for (size_t k = 0; k < N; k++) {
        float nrm = 0;
        for (size_t i = 0; i < M; i++) {
            nrm += A[i * N + k] * A[i * N + k];
        }
        R[k * N + k] = sqrt(nrm);
        for (size_t i = 0; i < M; i++) {
            Q[i * N + k] = A[i * N + k] / R[k * N + k];
        }
        for (size_t j = k + 1; j < N; j++) {
            R[k * N + j] = 0;
            for (size_t i = 0; i < M; i++) {
                R[k * N + j] += Q[i * N + k] * A[i * N + j];
            }
            for (size_t i = 0; i < M; i++) {
                A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
            }
        }
    }
}
