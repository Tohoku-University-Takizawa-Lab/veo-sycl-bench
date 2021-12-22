#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

#ifndef FLT_MAX
#define FLT_MAX 500000.0
#endif

using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    int nfeatures = 2;
    int nclusters = 3;
    std::vector<float> features(nfeatures * size, 2.0f);
    std::vector<float> clusters(nclusters * size, 1.0f);
    std::vector<int> membership(size, 0);

    buffer<float> features_buf(features.data(), range<1>(nfeatures * size));
    buffer<float> clusters_buf(clusters.data(), range<1>(nclusters * size));
    buffer<int> membership_buf(membership.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto features = features_buf.get_access<access::mode::read>(cgh);
        auto clusters = clusters_buf.get_access<access::mode::read>(cgh);
        auto membership = membership_buf.get_access<access::mode::discard_write>(cgh);
        cgh.parallel_for<class KmeansKernel>(range<1>(size),
        [features, clusters, membership, problem_size = size, nclusters_ = nclusters, nfeatures_ = nfeatures](id<1> idx) {
            size_t gid = idx[0];
            if (gid < problem_size) {
                int index = 0;
                float min_dist = FLT_MAX;
                for (size_t i = 0; i < nclusters_; i++) {
                    float dist = 0;
                    for (size_t l = 0; l < nfeatures_; l++) {
                        dist += (features[l * problem_size + gid] - clusters[i * nfeatures_ + l]) *
                                (features[l * problem_size + gid] - clusters[i * nfeatures_ + l]);
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        index = gid;
                    }
                }
                membership[gid] = index;
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "kmeans size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;

}
