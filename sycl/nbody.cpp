#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#define SOFTENING 1e-9f

REGISTER_KERNEL(bodyForce);
REGISTER_KERNEL(integratePos);

using namespace cl::sycl;

void randomizeBodies(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}
typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

int main(const int argc, const char** argv) {
    long begin = get_time();
    long nBodies = 5;
    if (argc > 1)
        nBodies = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);

    long bytes = nBodies * sizeof(Body);
    float* buf = (float*)malloc(bytes);
    Body* p = (Body*)buf;
    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data
    buffer<Body> body_buf(p, range<1>(nBodies));
    long start = get_time();
    q.submit([&](handler& cgh) {
        auto body_access =
            body_buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class bodyForce>(range<1>(nBodies), [=](id<1> i) {
            const float dt = 0.01f;
            float Fx = 0.0f;
            float Fy = 0.0f;
            float Fz = 0.0f;
            for (int j = 0; j < nBodies; j++) {
                float dx = body_access[j].x - body_access[i].x;
                float dy = body_access[j].y - body_access[i].y;
                float dz = body_access[j].z - body_access[i].z;
                float distSqr = dx * dx + dy * dy + dz * dz + 1e-9f;
                float invDist = 1.0f / sqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            body_access[i].vx += dt * Fx;
            body_access[i].vy += dt * Fy;
            body_access[i].vz += dt * Fz;
        });
    });
    q.wait();
    q.submit([&](handler& cgh) {
        auto body_access =
            body_buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class integratePos>(
            range<1>(nBodies), [=](id<1> i) {
                const float dt = 0.01f;
                body_access[i].x += body_access[i].vx * dt;
                body_access[i].y += body_access[i].vy * dt;
                body_access[i].z += body_access[i].vz * dt;
            });
    });
    q.wait();
    long end = get_time();
    long cost = end - start;
    printf("kernel time: %ld ms\n", cost);
    free(buf);
    printf("runtime %ld ms\n", get_time() - begin);
    return 0;
}