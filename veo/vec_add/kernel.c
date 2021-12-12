#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void kernel_add(int* input1, int* input2, int* output, size_t size) {
    // printf("add size=%d\n", size);
    for (size_t i = 0; i < size; ++i) {
        output[i] = input1[i] + input2[i];
    }
}
