#include <stdio.h>
#include <stdlib.h>

void conv2D(float* A, float* B, size_t size) {
    // printf("conv2D size=%d\n", size);
    const size_t NI = size;
    const size_t NJ = size;
    const float c11 = +0.2, c21 = +0.5, c31 = -0.8;
    const float c12 = -0.3, c22 = +0.6, c32 = -0.9;
    const float c13 = +0.4, c23 = +0.7, c33 = +0.10;
    for (size_t i = 1; i < NI - 1; ++i) {
        for (size_t j = 1; j < NJ - 1; ++j) {
            B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] + c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] + c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] + c33 * A[(i + 1) * NJ + (j + 1)];
        }
    }
}
