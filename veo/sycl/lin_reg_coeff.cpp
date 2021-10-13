#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(coeff);
using namespace cl::sycl;

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);

    long problem_bytes = size * sizeof(float);
    float* input1 = (float*)malloc(problem_bytes);
    float* input2 = (float*)malloc(problem_bytes);
    for (size_t i = 0; i < size; i++) {
        input1[i] = 1.0;
        input2[i] = 2.0;
    }
    float* sumv1 = (float*)malloc(sizeof(float));
    float* sumv2 = (float*)malloc(sizeof(float));
    float* xy = (float*)malloc(sizeof(float));
    float* xx = (float*)malloc(sizeof(float));
    *sumv1 = 0;
    *sumv2 = 0;
    *xy = 0;
    *xx = 0;
    buffer<float> in1_buff(input1, range<1>(size));
    buffer<float> in2_buff(input2, range<1>(size));
    buffer<float> s1_buff(sumv1, range<1>(1));
    buffer<float> s2_buff(sumv2, range<1>(1));
    buffer<float> xy_buff(xy, range<1>(1));
    buffer<float> xx_buff(xx, range<1>(1));

    q.submit([&](handler& cgh) {
        auto in1_access = in1_buff.get_access<access::mode::read>(cgh);
        auto in2_access = in2_buff.get_access<access::mode::read>(cgh);
        auto s1_access = s1_buff.get_access<access::mode::read_write>(cgh);
        auto s2_access = s2_buff.get_access<access::mode::read_write>(cgh);
        auto xy_access = xy_buff.get_access<access::mode::read_write>(cgh);
        auto xx_access = xx_buff.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class coeff>(range<1>(size), [=](id<1> i) {
            for (size_t i = 0; i < size; ++i) {
                s1_access[0] += input1[i];
                s2_access[0] += input2[i];
                xy[0] += input1[i] * input2[i];
                xx[0] += input1[i] * input1[i];
            }
        });
    });
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}