#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>


REGISTER_KERNEL(conv2D);
using namespace cl::sycl;

void init(DATA_TYPE* A, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;

    for (size_t i = 0; i < NI; ++i) {
        for (size_t j = 0; j < NJ; ++j) {
            A[i * NJ + j] = (DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX;
        }
    }
}

int main(const int argc, const char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    
    int problem_bytes = size * size * sizeof(DATA_TYPE);
    DATA_TYPE* A = (DATA_TYPE*)malloc(problem_bytes);
    init(A, size);
    buffer<DATA_TYPE> A_buff(A, range<1>(size * size));
    DATA_TYPE* B = (DATA_TYPE*)malloc(problem_bytes);
    buffer<DATA_TYPE> B_buff(B, range<1>(size * size));
    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto B_access = B_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class conv2D>(range<1>(size), [=](id<1> i) {
            const size_t NI = size;
            const size_t NJ = size;
            const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
            const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
            const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;
            for (size_t i = 1; i < NI - 1; ++i) {
                for (size_t j = 1; j < NJ - 1; ++j) {
                    B_access[i * NJ + j] = c11 * A_access[(i - 1) * NJ + (j - 1)] + c12 * A_access[(i + 0) * NJ + (j - 1)] + c13 * A_access[(i + 1) * NJ + (j - 1)] + c21 * A_access[(i - 1) * NJ + (j + 0)] + c22 * A_access[(i + 0) * NJ + (j + 0)] + c23 * A_access[(i + 1) * NJ + (j + 0)] + c31 * A_access[(i - 1) * NJ + (j + 1)] + c32 * A_access[(i + 0) * NJ + (j + 1)] + c33 * A_access[(i + 1) * NJ + (j + 1)];
                }
            }
        });
    });
    q.wait();
    free(A);
    printf("runtime %ld ms\n", get_time() - begin);
    return 0;
}