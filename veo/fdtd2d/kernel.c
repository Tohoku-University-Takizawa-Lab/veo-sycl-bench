#include <stdio.h>
#include <stdlib.h>

void kernel_Fdtd(float* fict, float* ex, float* ey, float* hz, size_t size) {
    // printf("fdtd size=%d\n", size);
    const size_t NX = size;
    const size_t NY = size;
    const int TMAX = 500;
    for (size_t t = 0; t < TMAX; t++) {
        for (size_t j = 0; j < NY; j++) {
            ey[0 * NY + j] = fict[t];
        }
        for (size_t i = 1; i < NX; i++) {
            for (size_t j = 0; j < NY; j++) {
                ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }
        for (size_t i = 0; i < NX; i++) {
            for (size_t j = 1; j < NY; j++) {
                ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }
        for (size_t i = 0; i < NX; i++) {
            for (size_t j = 0; j < NY; j++) {
                hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
            }
        }
    }
}
