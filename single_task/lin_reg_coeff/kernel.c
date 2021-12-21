#include <stdio.h>
#include <stdlib.h>

void P5coeff(float* input1, float* input2, float* sumv1, float* sumv2, float* xy, float* xx, int *size) {
    // printf("coeff *size=%d\n", *size);
    for (int i = 0; i < *size; ++i) {
        *sumv1 += input1[i];
        *sumv2 += input2[i];
        *xy += input1[i] * input2[i];
        *xx += input1[i] * input1[i];
    }
    // printf("ve result:sumv1=%f sumv2=%f xy=%f xx=%f\n", *sumv1, *sumv2, *xy, *xx);
}
