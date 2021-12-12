#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void P3add(int* input1, int* input2, int* output, int* size) {
    // printf("add size = %d\n", *size);
    #pragma omp parallel for
    for (int i = 0; i < *size; ++i) {
        output[i] = input1[i] + input2[i];
    }
    // int tid, nthreads = 0;
    // #pragma omp parallel private(nthreads, tid)
    // {
    //     tid = omp_get_thread_num();
    //     printf("Hello, World! from thread = %d\n", tid);
    //     if (tid == 0) {
    //         nthreads = omp_get_num_threads();
    //         printf("Number of threads = %d\n", nthreads);
    //     }
    // } /* All threads join master thread and disband */
    // fflush(stdout);
}
