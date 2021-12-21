#include <stdio.h>
#include <stdlib.h>

float float_n = 3214212.01;
void P10covariance(float* data, float* symmat, float* mean, int* size) {
    // printf("covariance *size=%d\n", *size);
    const auto M = *size;
    const auto N = *size;
    for (int j = 1; j <= M; j++) {
        mean[j] = 0.0;
        for (int i = 1; i <= N; i++) {
            mean[j] += data[i * (M + 1) + j];
        }
        mean[j] /= float_n;
    }
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            data[i * (M + 1) + j] -= mean[j];
        }
    }
    for (int j1 = 1; j1 <= M; j1++) {
        for (int j2 = j1; j2 <= M; j2++) {
            symmat[j1 * (M + 1) + j2] = 0.0;
            for (int i = 1; i <= N; i++) {
                symmat[j1 * (M + 1) + j2] += data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
            }
            symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
        }
    }
}
