#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(mm2);
using namespace cl::sycl;

/* Array initialization. */
void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    const size_t NL = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NK; j++) {
            A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
        }
    }
    for (size_t i = 0; i < NK; i++) {
        for (size_t j = 0; j < NJ; j++) {
            B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
        }
    }
    for (size_t i = 0; i < NL; i++) {
        for (size_t j = 0; j < NJ; j++) {
            C[i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
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
    DATA_TYPE* D = (DATA_TYPE*)malloc(problem_bytes);
    init_array(A, B, C, D, size);
    DATA_TYPE* E = (DATA_TYPE*)malloc(problem_bytes);
    buffer<DATA_TYPE> A_buff(A, range<1>(size * size));
    buffer<DATA_TYPE> B_buff(B, range<1>(size * size));
    buffer<DATA_TYPE> C_buff(C, range<1>(size * size));
    buffer<DATA_TYPE> D_buff(D, range<1>(size * size));
    buffer<DATA_TYPE> E_buff(E, range<1>(size * size));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto B_access = B_buff.get_access<access::mode::read>(cgh);
        auto C_access = C_buff.get_access<access::mode::read_write>(cgh);
        auto D_access = D_buff.get_access<access::mode::read>(cgh);
        auto E_access = E_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class mm2>(range<1>(size), [=](id<1> i) {
            const size_t NI = size;
            const size_t NJ = size;
            const size_t NK = size;
            const size_t NL = size;
            for (size_t i = 0; i < NI; i++) {
                for (size_t j = 0; j < NJ; j++) {
                    for (size_t k = 0; k < NK; ++k) {
                        C_access[i * NJ + j] += A_access[i * NK + k] * B_access[k * NJ + j];
                    }
                }
            }
            for (size_t i = 0; i < NI; i++) {
                for (size_t j = 0; j < NL; j++) {
                    E_access[i * NL + j] = 0;
                    for (size_t k = 0; k < NJ; ++k) {
                        E_access[i * NL + j] += C_access[i * NJ + k] * D_access[k * NL + j];
                        // printf("%f ", E[i * NL + j]);
                    }
                }
            }
        });
    });
    q.wait();
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}