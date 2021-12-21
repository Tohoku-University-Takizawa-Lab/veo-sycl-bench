#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(kmeans);
using namespace cl::sycl;
#define FLT_MAX 500000.0

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    int nfeatures = 2;
    int nclusters = 3;
    long feature_size = nfeatures * size * sizeof(float);
    long cluster_size = nclusters * size * sizeof(float);
    std::vector<float> features(nfeatures * size, 2.0f);
    std::vector<float> clusters(nclusters * size, 1.0f);
    std::vector<int> membership(size);
    buffer<float> feature_buff(features.data(), range<1>(nfeatures * size));
    buffer<float> cluster_buff(clusters.data(), range<1>(nclusters * size));
    buffer<int> m_buff(membership.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto f_access = feature_buff.get_access<access::mode::read>(cgh);
        auto c_access = cluster_buff.get_access<access::mode::read>(cgh);
        auto m_access = m_buff.get_access<access::mode::write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class kmeans>([=]() {
            for (int gid = 0; gid < no_access[0]; ++gid) {
                int index = 0;
                double min_dist = FLT_MAX;
                for (int i = 0; i < nclusters; i++) {
                    double dist = 0;
                    for (int l = 0; l < nfeatures; l++) {
                        dist +=
                            (f_access[l * no_access[0] + gid] - c_access[i * nfeatures + l]) *
                            (f_access[l * no_access[0] + gid] - c_access[i * nfeatures + l]);
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
    auto end = std::chrono::steady_clock::now();
    std::cout << "kmeans size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}