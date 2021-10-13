#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(correlation);
using namespace cl::sycl;

#define FLOAT_N 3214212.01
#define EPS 0.005
#define sqrt_of_array_cell(x, j) sqrt(x[j])

void init_arrays(DATA_TYPE* data_access, size_t size) {
    const size_t M = size;
    const size_t N = size;

    for (size_t i = 0; i <= M; i++) {
        for (size_t j = 0; j <= N; j++) {
            data_access[i * N + j] = ((DATA_TYPE)i * j) / (M + 1);
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
    DATA_TYPE* stddev = (DATA_TYPE*)malloc((size + 1) * sizeof(DATA_TYPE));
    DATA_TYPE* symmat = (DATA_TYPE*)malloc((size + 1) * (size + 1) * sizeof(DATA_TYPE));

    buffer<DATA_TYPE> data_buff(data, range<1>((size + 1) * (size + 1)));
    buffer<DATA_TYPE> mean_buff(mean, range<1>(size + 1));
    buffer<DATA_TYPE> stddev_buff(stddev, range<1>(size + 1));
    buffer<DATA_TYPE> symmat_buff(symmat, range<1>((size + 1) * (size + 1)));

    q.submit([&](handler& cgh) {
        auto data_access = data_buff.get_access<access::mode::read_write>(cgh);
        auto mean_access = mean_buff.get_access<access::mode::write>(cgh);
        auto stddev_access = stddev_buff.get_access<access::mode::write>(cgh);
        auto symmat_access = symmat_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class correlation>(range<1>(size), [=](id<1> i) {
            const size_t M = size;
            const size_t N = size;
            for (size_t j = 1; j <= M; j++) {
                mean_access[j] = 0.0;
                for (size_t i = 1; i <= N; i++) {
                    mean_access[j] += data_access[i * (M + 1) + j];
                }
                mean_access[j] /= (DATA_TYPE)FLOAT_N;
            }
            for (size_t j = 1; j <= M; j++) {
                stddev_access[j] = 0.0;
                for (size_t i = 1; i <= N; i++) {
                    stddev_access[j] += (data_access[i * (M + 1) + j] - mean_access[j]) * (data_access[i * (M + 1) + j] - mean_access[j]);
                }
                stddev_access[j] /= FLOAT_N;
                stddev_access[j] = sqrt_of_array_cell(stddev_access, j);
                stddev_access[j] = stddev_access[j] <= EPS ? 1.0 : stddev_access[j];
            }
            for (size_t i = 1; i <= N; i++) {
                for (size_t j = 1; j <= M; j++) {
                    data_access[i * (M + 1) + j] -= mean_access[j];
                    data_access[i * (M + 1) + j] /= sqrt(FLOAT_N);
                    data_access[i * (M + 1) + j] /= stddev_access[j];
                }
            }
            for (size_t j1 = 1; j1 <= M - 1; j1++) {
                symmat_access[j1 * (M + 1) + j1] = 1.0;
                for (size_t j2 = j1 + 1; j2 <= M; j2++) {
                    symmat_access[j1 * (M + 1) + j2] = 0.0;
                    for (size_t i = 1; i <= N; i++) {
                        symmat_access[j1 * (M + 1) + j2] += (data_access[i * (M + 1) + j1] * data_access[i * (M + 1) + j2]);
                    }
                    symmat_access[j2 * (M + 1) + j1] = symmat_access[j1 * (M + 1) + j2];
                }
            }
            symmat_access[M * (M + 1) + M] = 1.0;
        });
    });
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}