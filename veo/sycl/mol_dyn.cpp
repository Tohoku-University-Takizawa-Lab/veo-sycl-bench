#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(dyn);
using namespace cl::sycl;

typedef struct {
    float x, y, z;
} Atom;

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);

    long problem_bytes = size * sizeof(Atom);
    float* buf = (float*)malloc(problem_bytes);
    Atom* input = (Atom*)buf;
    Atom* output = (Atom*)malloc(problem_bytes);
    int* neighbour = (int*)malloc(size * sizeof(int));
    for (size_t i = 0; i < size; i++) {
        Atom temp = {(float)i, (float)i, (float)i};
        input[i] = temp;
    }
    for (size_t i = 0; i < size; i++) {
        neighbour[i] = i + 1;
    }

    buffer<Atom> in_buff(input, range<1>(size));
    buffer<int> n_buff(neighbour, range<1>(size));
    buffer<Atom> out_buff(output, range<1>(size));

    q.submit([&](handler& cgh) {
        auto in_access = in_buff.get_access<access::mode::read>(cgh);
        auto out_access = out_buff.get_access<access::mode::read_write>(cgh);
        auto n_access = n_buff.get_access<access::mode::read>(cgh);
        cgh.parallel_for<class dyn>(range<1>(size), [=](id<1> i) {
            int neighCount = 15;
            int cutsq = 50;
            int lj1 = 20;
            float lj2 = 0.003f;
            int inum = 0;
            for (int i = 0; i < size; ++i) {
                Atom ipos = in_access[i];
                Atom f = {0.0f, 0.0f, 0.0f};
                int j = 0;
                while (j < neighCount) {
                    int jidx = n_access[j * inum + i];
                    Atom jpos = in_access[jidx];
                    float delx = ipos.x - jpos.x;
                    float dely = ipos.y - jpos.y;
                    float delz = ipos.z - jpos.z;
                    float r2inv = delx * delx + dely * dely + delz * delz;
                    if (r2inv < cutsq) {
                        r2inv = 10.0f / r2inv;
                        float r6inv = r2inv * r2inv * r2inv;
                        float forceC = r2inv * r6inv * (lj1 * r6inv - lj2);
                        f.x += delx * forceC;
                        f.y += dely * forceC;
                        f.z += delz * forceC;
                    }
                    j++;
                }
                out_access[i] = f;
            }
        });
    });
    q.wait();
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}