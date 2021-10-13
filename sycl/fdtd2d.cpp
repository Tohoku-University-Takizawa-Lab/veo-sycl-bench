#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(fdtd2d);
using namespace cl::sycl;

int TMAX = 500;
void init_arrays(DATA_TYPE* fict, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t size) {
    const size_t NX = size;
    const size_t NY = size;

    for (size_t i = 0; i < TMAX; i++) {
        fict[i] = (DATA_TYPE)i;
    }

    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
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

    DATA_TYPE* fict = (DATA_TYPE*)malloc(TMAX * sizeof(DATA_TYPE));
    DATA_TYPE* ex = (DATA_TYPE*)malloc((size + 1) * size * sizeof(DATA_TYPE));
    DATA_TYPE* ey = (DATA_TYPE*)malloc((size + 1) * size * sizeof(DATA_TYPE));
    DATA_TYPE* hz = (DATA_TYPE*)malloc(size * size * sizeof(DATA_TYPE));
    init_arrays(fict, ex, ey, hz, size);

    buffer<DATA_TYPE> fict_buff(fict, range<1>(TMAX));
    buffer<DATA_TYPE> ex_buff(ex, range<1>((size + 1) * size));
    buffer<DATA_TYPE> ey_buff(ey, range<1>((size + 1) * size));
    buffer<DATA_TYPE> hz_buff(hz, range<1>(size * size));

    q.submit([&](handler& cgh) {
        auto fict_access = fict_buff.get_access<access::mode::read_write>(cgh);
        auto ex_access = ex_buff.get_access<access::mode::read_write>(cgh);
        auto ey_access = ey_buff.get_access<access::mode::read_write>(cgh);
        auto hz_access = hz_buff.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class fdtd2d>(range<1>(size), [=](id<1> i) {
            const size_t NX = size;
            const size_t NY = size;
            const int TMAX = 500;
            for (size_t t = 0; t < TMAX; t++) {
                for (size_t j = 0; j < NY; j++) {
                    ey_access[0 * NY + j] = fict_access[t];
                }
                for (size_t i = 1; i < NX; i++) {
                    for (size_t j = 0; j < NY; j++) {
                        ey_access[i * NY + j] = ey_access[i * NY + j] - 0.5 * (hz_access[i * NY + j] - hz_access[(i - 1) * NY + j]);
                    }
                }
                for (size_t i = 0; i < NX; i++) {
                    for (size_t j = 1; j < NY; j++) {
                        ex_access[i * (NY + 1) + j] = ex_access[i * (NY + 1) + j] - 0.5 * (hz_access[i * NY + j] - hz_access[i * NY + (j - 1)]);
                    }
                }
                for (size_t i = 0; i < NX; i++) {
                    for (size_t j = 0; j < NY; j++) {
                        hz_access[i * NY + j] = hz_access[i * NY + j] - 0.7 * (ex_access[i * (NY + 1) + (j + 1)] - ex_access[i * (NY + 1) + j] + ey_access[(i + 1) * NY + j] - ey_access[i * NY + j]);
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