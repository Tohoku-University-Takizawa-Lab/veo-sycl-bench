#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(add);
using namespace cl::sycl;

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);

    long problem_bytes = size * sizeof(DATA_TYPE);
    DATA_TYPE* input1 = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* input2 = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* output = (DATA_TYPE*)malloc(problem_bytes);
    for (size_t i = 0; i < size; i++) {
        input1[i] = (DATA_TYPE)i;
        input2[i] = (DATA_TYPE)i;
        output[i] = 0;
    }

    buffer<DATA_TYPE> in1_buff(input1, range<1>(size));
    buffer<DATA_TYPE> in2_buff(input2, range<1>(size));
    buffer<DATA_TYPE> out_buff(output, range<1>(size));

    q.submit([&](handler& cgh) {
        auto in1_access = in1_buff.get_access<access::mode::read>(cgh);
        auto in2_access = in2_buff.get_access<access::mode::read>(cgh);
        auto out_access = out_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class add>(range<1>(size), [=](id<1> i) {
            for (size_t i = 0; i < size; ++i) {
                out_access[i] = in1_access[i] + in2_access[i];
            }
        });
    });
    q.wait();
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}