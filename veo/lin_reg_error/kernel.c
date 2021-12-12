#include <stdio.h>
#include <stdlib.h>

void kernel_error(float* input1, float* input2, float* alpha, float* beta, float* output, size_t size) {
    // printf("lin_reg_error size=%d\n", size);
    for (size_t i = 0; i < size; i++) {
        float error = 0.0;
        for (size_t j = 0; j < size; j++) {
            float e = (alpha[i] * input1[j] + beta[i]) - input2[j];
            error += e * e;
        }
        output[i] = error;
    }
}
