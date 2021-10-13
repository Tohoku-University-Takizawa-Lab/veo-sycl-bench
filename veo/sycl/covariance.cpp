#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(covariance);
using namespace cl::sycl;

void init_arrays(DATA_TYPE* data, size_t size) {
    const size_t M = size;
    const size_t N = size;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
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

    DATA_TYPE* data = (DATA_TYPE*)malloc((size + 1) * (size + 1) * sizeof(DATA_TYPE));
    init_arrays(data, size);
    DATA_TYPE* mean = (DATA_TYPE*)malloc((size + 1) * sizeof(DATA_TYPE));
    DATA_TYPE* symmat = (DATA_TYPE*)malloc((size + 1) * (size + 1) * sizeof(DATA_TYPE));

    buffer<DATA_TYPE> data_buff(data, range<1>((size + 1) * (size + 1)));
    buffer<DATA_TYPE> mean_buff(mean, range<1>(size + 1));
    buffer<DATA_TYPE> symmat_buff(symmat, range<1>((size + 1) * (size + 1)));

    q.submit([&](handler& cgh) {
        auto data_access = data_buff.get_access<access::mode::write>(cgh);
        auto symmat_access = symmat_buff.get_access<access::mode::write>(cgh);
        auto mean_access = mean_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class covariance>(range<1>(size), [=](id<1> i) {
            DATA_TYPE float_n = 3214212.01;
            const auto M = size;
            const auto N = size;
            for (size_t j = 1; j <= M; j++) {
                mean_access[j] = 0.0;
                for (size_t i = 1; i <= N; i++) {
                    mean_access[j] += data_access[i * (M + 1) + j];
                }
                mean_access[j] /= float_n;
            }
            for (size_t i = 1; i <= N; i++) {
                for (size_t j = 1; j <= M; j++) {
                    data_access[i * (M + 1) + j] -= mean_access[j];
                }
            }
            for (size_t j1 = 1; j1 <= M; j1++) {
                for (size_t j2 = j1; j2 <= M; j2++) {
                    symmat_access[j1 * (M + 1) + j2] = 0.0;
                    for (size_t i = 1; i <= N; i++) {
                        symmat_access[j1 * (M + 1) + j2] += data_access[i * (M + 1) + j1] * data_access[i * (M + 1) + j2];
                    }
                    symmat_access[j2 * (M + 1) + j1] = symmat_access[j1 * (M + 1) + j2];
                }
            }
        });
    });
    q.wait();
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}