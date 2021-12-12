#include <stdio.h>
#include <stdlib.h>

void kernel_prod(float* input1, float* input2, float* output, size_t size) {
    // printf("prod size=%d\n", size);
    for (size_t i = 0; i < size; ++i) {
        output[i] = input1[i] * input2[i];
    }
}
