#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(atax);
using namespace cl::sycl;

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(DATA_TYPE* x, DATA_TYPE* A, size_t size) {
    const size_t NX = size;
    const size_t NY = size;
    for (size_t i = 0; i < NX; i++) {
        x[i] = i * M_PI;
        for (size_t j = 0; j < NY; j++) {
            A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
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
    DATA_TYPE* x = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* A = (DATA_TYPE*)malloc(problem_bytes);
    init_array(x, A, size);

    DATA_TYPE* y = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* tmp = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));

    buffer<DATA_TYPE> A_buff(A, range<1>(size * size));
    buffer<DATA_TYPE> x_buff(x, range<1>(size));
    buffer<DATA_TYPE> y_buff(y, range<1>(size));
    buffer<DATA_TYPE> tmp_buff(tmp, range<1>(size));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto x_access = x_buff.get_access<access::mode::read>(cgh);
        auto y_access = y_buff.get_access<access::mode::write>(cgh);
        auto tmp_access = tmp_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class atax>(range<1>(size), [=](id<1> i) {
            const size_t NX = size;
            const size_t NY = size;
            for (size_t i = 0; i < NX; i++) {
                for (size_t j = 0; j < NY; j++) {
                    tmp_access[i] += A_access[i * NY + j] * x[j];
                }
                for (size_t j = 0; j < NY; j++) {
                    y_access[j] += A_access[i * NY + j] * tmp_access[i];
                }
            }
        });
    });
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}