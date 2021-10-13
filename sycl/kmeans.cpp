#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

REGISTER_KERNEL(kmeans);
using namespace cl::sycl;
#define FLT_MAX 500000.0

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);

    int nfeatures = 2;
    int nclusters = 3;
    long feature_size = nfeatures * size * sizeof(float);
    long cluster_size = nclusters * size * sizeof(float);
    float* features = (float*)malloc(feature_size);
    float* clusters = (float*)malloc(cluster_size);
    for (int i = 0; i < nfeatures * size; ++i) {
        features[i] = 2.0f;
    }
    for (int i = 0; i < nclusters * size; ++i) {
        clusters[i] = 1.0f;
    }
    int* membership = (int*)malloc(size * sizeof(int));

    buffer<float> feature_buff(features, range<1>(nfeatures * size));
    buffer<float> cluster_buff(clusters, range<1>(nclusters * size));
    buffer<int> m_buff(membership, range<1>(size));
    q.submit([&](handler& cgh) {
        auto f_access = feature_buff.get_access<access::mode::read>(cgh);
        auto c_access = cluster_buff.get_access<access::mode::read>(cgh);
        auto m_access = m_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class kmeans>(range<1>(size), [=](id<1> i) {
            for (size_t gid = 0; gid < size; ++gid) {
                int index = 0;
                double min_dist = FLT_MAX;
                for (size_t i = 0; i < nclusters; i++) {
                    double dist = 0;
                    for (size_t l = 0; l < nfeatures; l++) {
                        dist +=
                            (f_access[l * size + gid] - c_access[i * nfeatures + l]) *
                            (f_access[l * size + gid] - c_access[i * nfeatures + l]);
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        index = gid;
                    }
                }
                m_access[gid] = index;
            }
        });
    });
    q.wait();
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}