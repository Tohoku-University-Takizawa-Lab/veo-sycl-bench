#include <stdio.h>
#include <stdlib.h>

void P4prod(float* input1, float* input2, float* output, int *size) {
    // printf("prod *size=%d\n", *size);
    for (int i = 0; i < *size; ++i) {
        output[i] = input1[i] * input2[i];
    }
}
