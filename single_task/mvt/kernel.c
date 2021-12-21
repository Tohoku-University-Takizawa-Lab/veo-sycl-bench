#include <stdio.h>
#include <stdlib.h>

void P3mvt(float* a, float* x1, float* x2, float* y1, float* y2, int *size) {
    // printf("mvt *size=%d\n", *size);
    const int N = *size;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x1[i] = x1[i] + a[i * N + j] * y1[j];
        }
    }
    for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
            x2[k] = x2[k] + a[k * N + l] * y2[l];
        }
    }
}
