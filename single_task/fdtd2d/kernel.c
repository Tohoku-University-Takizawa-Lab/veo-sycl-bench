#include <stdio.h>
#include <stdlib.h>

void P6fdtd2d(float* fict, float* ex, float* ey, float* hz, int *size) {
    // printf("fdtd *size=%d\n", *size);
    const int NX = *size;
    const int NY = *size;
    const int TMAX = 500;
    for (int t = 0; t < TMAX; t++) {
        for (int j = 0; j < NY; j++) {
            ey[0 * NY + j] = fict[t];
        }
        for (int i = 1; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }
        for (int i = 0; i < NX; i++) {
            for (int j = 1; j < NY; j++) {
                ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
            }
        }
    }
}
