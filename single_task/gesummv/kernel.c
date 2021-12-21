#include <stdio.h>
#include <stdlib.h>

void P7gesummv(float* A, float* B, float* x, float* y, float* tmp, int* size) {
    // printf("gesummv *size=%d\n", *size);
    const int N = *size;
    float alpha = 1;
    float beta = 1;
    for (int i = 0; i < N; i++) {
        tmp[i] = 0;
        y[i] = 0;
        for (int j = 0; j < N; j++) {
            tmp[i] = A[i * N + j] * x[j] + tmp[i];
            y[i] = B[i * N + j] * x[j] + y[i];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}
