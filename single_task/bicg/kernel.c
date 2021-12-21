#include <stdio.h>
#include <stdlib.h>

void P4bicg(float* A, float* r, float* s, float* p, float* q, int *size) {
    // printf("bicg *size=%d\n", *size);
    const int NX = *size;
    const int NY = *size;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            s[j] += r[i] * A[i * NY + j];
            q[i] += A[i * NY + j] * p[j];
        }
    }
}
