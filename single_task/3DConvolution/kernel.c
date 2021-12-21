#include <stdio.h>
#include <stdlib.h>

void P6conv3D(float* A, float* B, int *size) {
    // printf("conv3D *size=%d\n", *size);
    const int NI = *size;
    const int NJ = *size;
    const int NK = *size;
    const float c11 = +2, c21 = +5, c31 = -8;
    const float c12 = -3, c22 = +6, c32 = -9;
    const float c13 = +4, c23 = +7, c33 = +10;
    for (int i = 1; i < NI - 1; ++i) {
        for (int j = 1; j < NJ - 1; ++j) {
            for (int k = 1; k < NK - 1; ++k) {
                B[i * (NK * NJ) + j * NK + k] = c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] + c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] + c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
            }
        }
    }
}
