#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
#include <vector>

REGISTER_KERNEL(error);
using namespace cl::sycl;
using namespace std;

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;
    vector<float> input1;
    vector<float> input2;
    vector<float> alpha;
    vector<float> beta;
    input1.resize(size);
    input2.resize(size);
    alpha.resize(size);
    beta.resize(size);
    for (size_t i = 0; i < size; i++) {
        input1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        input2[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        alpha[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        beta[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    vector<float> output;
    output.resize(size, 0);
    buffer<float> input1_buf(input1.data(), range<1>(size));
    buffer<float> input2_buf(input2.data(), range<1>(size));
    buffer<float> alpha_buf(alpha.data(), range<1>(size));
    buffer<float> beta_buf(beta.data(), range<1>(size));
    buffer<float> output_buf(output.data(), range<1>(size));
    q.submit([&](handler& cgh) {
        auto in1 = input1_buf.get_access<access::mode::read>(cgh);
        auto in2 = input2_buf.get_access<access::mode::read>(cgh);
        auto al = alpha_buf.get_access<access::mode::read>(cgh);
        auto be = beta_buf.get_access<access::mode::read>(cgh);
        auto out = output_buf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<class error>(range<1>(size), [=](id<1> idx) {
            size_t gid = idx[0];
            float a = al[gid];
            float b = be[gid];
            float error = 0.0;
            if (gid < size) {
                for (size_t i = 0; i < size; i++) {
                    float e = (a * in1[i] + b) - in2[i];
                    error += e * e;
                }
            }
            out[gid] = error;
        });
    });
    q.wait();
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}


