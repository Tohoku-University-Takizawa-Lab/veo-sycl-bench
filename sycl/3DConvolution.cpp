#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(conv3D);
using namespace cl::sycl;

void init(DATA_TYPE* A, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    for (size_t i = 0; i < NI; ++i) {
        for (size_t j = 0; j < NJ; ++j) {
            for (size_t k = 0; k < NK; ++k) {
                A[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
            }
        }
    }
}

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);

    int problem_bytes = size * size * size * sizeof(DATA_TYPE);
    DATA_TYPE* A = (DATA_TYPE*)malloc(problem_bytes);
    init(A, size);
    DATA_TYPE* B = (DATA_TYPE*)malloc(problem_bytes);

    buffer<DATA_TYPE> A_buff(A, range<1>(size * size * size));
    buffer<DATA_TYPE> B_buff(B, range<1>(size * size * size));
    
    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto B_access = B_buff.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class conv3D>(range<1>(size), [=](id<1> i) {
            const size_t NI = size;
            const size_t NJ = size;
            const size_t NK = size;
            const DATA_TYPE c11 = +2, c21 = +5, c31 = -8;
            const DATA_TYPE c12 = -3, c22 = +6, c32 = -9;
            const DATA_TYPE c13 = +4, c23 = +7, c33 = +10;
            for (size_t i = 1; i < NI - 1; ++i) {
                for (size_t j = 1; j < NJ - 1; ++j) {
                    for (size_t k = 1; k < NK - 1; ++k) {
                        B_access[i * (NK * NJ) + j * NK + k] = c11 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c21 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c31 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c12 * A_access[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A_access[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] + c32 * A_access[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c13 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A_access[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c23 * A_access[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A_access[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] + c33 * A_access[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
                    }
                }
            }
        });
    });
    q.wait();
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}