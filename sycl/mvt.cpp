#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(mvt);
using namespace cl::sycl;

void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2, size_t size) {
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        x1[i] = 0.0;
        x2[i] = 0.0;
        y_1[i] = 0.0;
        y_2[i] = 0.0;
        for (size_t j = 0; j < N; j++) {
            a[i * N + j] = (DATA_TYPE)(i + j + 1.0) / N;
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

    DATA_TYPE* a = (DATA_TYPE*)malloc(size * size * sizeof(DATA_TYPE));
    DATA_TYPE* x1 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* x2 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* y1 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* y2 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    init_arrays(a, x1, x2, y1, y2, size);

    buffer<DATA_TYPE> a_buff(a, range<1>(size * size));
    buffer<DATA_TYPE> x1_buff(x1, range<1>(size));
    buffer<DATA_TYPE> x2_buff(x2, range<1>(size));
    buffer<DATA_TYPE> y1_buff(y1, range<1>(size));
    buffer<DATA_TYPE> y2_buff(y2, range<1>(size));

    q.submit([&](handler& cgh) {
        auto a_access = a_buff.get_access<access::mode::read>(cgh);
        auto x1_access = x1_buff.get_access<access::mode::read_write>(cgh);
        auto x2_access = x2_buff.get_access<access::mode::read_write>(cgh);
        auto y1_access = y1_buff.get_access<access::mode::read>(cgh);
        auto y2_access = y2_buff.get_access<access::mode::read>(cgh);
        cgh.parallel_for<class mvt>(range<1>(size), [=](id<1> i) {
            const size_t N = size;
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    x1[i] = x1[i] + a[i * N + j] * y1[j];
                }
            }
            for (size_t k = 0; k < N; k++) {
                for (size_t l = 0; l < N; l++) {
                    x2[k] = x2[k] + a[k * N + l] * y2[l];
                }
            }
        });
    });
    q.wait();

    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}