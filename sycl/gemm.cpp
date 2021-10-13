#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(gemm);
using namespace cl::sycl;

#define ALPHA 32412
#define BETA 2123

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NK; j++) {
            A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
        }
    }
    for (size_t i = 0; i < NK; i++) {
        for (size_t j = 0; j < NJ; j++) {
            B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
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

    int problem_bytes = size * size * sizeof(DATA_TYPE);
    DATA_TYPE* A = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* B = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* C = (DATA_TYPE*)malloc(problem_bytes);
    init(A, B, C, size);

    buffer<DATA_TYPE> A_buff(A, range<1>(size * size));
    buffer<DATA_TYPE> B_buff(B, range<1>(size * size));
    buffer<DATA_TYPE> C_buff(C, range<1>(size * size));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto B_access = B_buff.get_access<access::mode::read>(cgh);
        auto C_access = C_buff.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class gemm>(range<1>(size), [=](id<1> i) {
            const size_t NI = size;
            const size_t NJ = size;
            const size_t NK = size;
            for (size_t i = 0; i < NI; i++) {
                for (size_t j = 0; j < NJ; j++) {
                    C_access[i * NJ + j] *= BETA;
                    for (size_t k = 0; k < NK; ++k) {
                        C_access[i * NJ + j] += ALPHA * A_access[i * NK + k] * B_access[k * NJ + j];
                    }
                }
            }
        });
    });
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}