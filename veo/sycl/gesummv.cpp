#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(gesummv);
using namespace cl::sycl;

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, size_t size) {
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        x[i] = 1;
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = 2;
            B[i * N + j] = 3;
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
    DATA_TYPE* x = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    init(A, B, x, size);
    DATA_TYPE* y = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* tmp = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));

    buffer<DATA_TYPE> A_buff(A, range<1>(size * size));
    buffer<DATA_TYPE> B_buff(A, range<1>(size * size));
    buffer<DATA_TYPE> x_buff(x, range<1>(size));
    buffer<DATA_TYPE> y_buff(y, range<1>(size));
    buffer<DATA_TYPE> tmp_buff(tmp, range<1>(size));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto B_access = B_buff.get_access<access::mode::read>(cgh);
        auto x_access = x_buff.get_access<access::mode::read>(cgh);
        auto y_access = y_buff.get_access<access::mode::read_write>(cgh);
        auto tmp_access = tmp_buff.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class gesummv>(range<1>(size), [=](id<1> i) {
            const size_t N = size;
            DATA_TYPE alpha = 1;
            DATA_TYPE beta = 1;
            for (size_t i = 0; i < N; i++) {
                tmp_access[i] = 0;
                y_access[i] = 0;
                for (size_t j = 0; j < N; j++) {
                    tmp_access[i] = A_access[i * N + j] * x_access[j] + tmp_access[i];
                    y_access[i] = B_access[i * N + j] * x_access[j] + y_access[i];
                }
                y_access[i] = alpha * tmp_access[i] + beta * y_access[i];
            }
        });
    });
    q.wait();

    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}