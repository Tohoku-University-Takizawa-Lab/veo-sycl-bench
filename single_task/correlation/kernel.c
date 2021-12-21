#include <stdio.h>
#include <stdlib.h>

#define FLOAT_N 3214212.01
#define EPS 0.005
#define sqrt_of_array_cell(x, j) sqrt(x[j])

void P11correlation(float* data, float* mean, float* stddev, float* symmat, int *size) {
    // printf("correlation *size=%d\n", *size);
    const int M = *size;
    const int N = *size;
    for (int j = 1; j <= M; j++) {
        mean[j] = 0.0;
        for (int i = 1; i <= N; i++) {
            mean[j] += data[i * (M + 1) + j];
        }
        mean[j] /= (float)FLOAT_N;
    }
    for (int j = 1; j <= M; j++) {
        stddev[j] = 0.0;
        for (int i = 1; i <= N; i++) {
            stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
        }
        stddev[j] /= FLOAT_N;
        stddev[j] = sqrt_of_array_cell(stddev, j);
        stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
    }
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            data[i * (M + 1) + j] -= mean[j];
            data[i * (M + 1) + j] /= sqrt(FLOAT_N);
            data[i * (M + 1) + j] /= stddev[j];
        }
    }
    for (int j1 = 1; j1 <= M - 1; j1++) {
        symmat[j1 * (M + 1) + j1] = 1.0;
        for (int j2 = j1 + 1; j2 <= M; j2++) {
            symmat[j1 * (M + 1) + j2] = 0.0;
            for (int i = 1; i <= N; i++) {
                symmat[j1 * (M + 1) + j2] += (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
            }
            symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
        }
    }
    symmat[M * (M + 1) + M] = 1.0;
}
