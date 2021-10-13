#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(bicg);
using namespace cl::sycl;

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* r, size_t size) {
    const size_t NX = size;
    const size_t NY = size;
    for (size_t i = 0; i < NX; i++) {
        r[i] = i * M_PI;

        for (size_t j = 0; j < NY; j++) {
            A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
        }
    }
    for (size_t i = 0; i < NY; i++) {
        p[i] = i * M_PI;
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
    DATA_TYPE* p = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* r = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    init_array(A, p, r, size);

    DATA_TYPE* s = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* Q = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));

    buffer<DATA_TYPE> A_buff(A, range<1>(size * size));
    buffer<DATA_TYPE> p_buff(p, range<1>(size));
    buffer<DATA_TYPE> r_buff(r, range<1>(size));
    buffer<DATA_TYPE> s_buff(s, range<1>(size));
    buffer<DATA_TYPE> Q_buff(Q, range<1>(size));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto r_access = r_buff.get_access<access::mode::read>(cgh);
        auto s_access = s_buff.get_access<access::mode::write>(cgh);
        auto p_access = p_buff.get_access<access::mode::read>(cgh);
        auto Q_access = Q_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class bicg>(range<1>(size), [=](id<1> i) {
            const size_t NX = size;
            const size_t NY = size;
            for (size_t i = 0; i < NX; i++) {
                for (size_t j = 0; j < NY; j++) {
                    s_access[j] += r_access[i] * A_access[i * NY + j];
                    Q_access[i] += A_access[i * NY + j] * p_access[j];
                }
            }
        });
    });
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}