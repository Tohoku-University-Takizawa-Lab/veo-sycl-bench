#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    int neighCount = 15;
    int cutsq = 50;
    int lj1 = 20;
    float lj2 = 0.003f;
    int inum = 0;
    std::vector<float4> input(size);
    std::vector<float4> output(size);
    std::vector<int> neighbour(size);
    for (size_t i = 0; i < size; i++) {
        input[i] = float4{(float)i, (float)i, (float)i, (float)i}; // Same value for all 4 elements. Could be changed if needed
    }
    for (size_t i = 0; i < size; i++) {
        neighbour[i] = (i + 1) % size;
    }
    buffer<float4> input_buf(input.data(), range<1>(size));
    buffer<float4> output_buf(output.data(), range<1>(size));
    buffer<int> neighbour_buf(neighbour.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto in = input_buf.get_access<access::mode::read>(cgh);
        auto neigh = neighbour_buf.get_access<access::mode::read>(cgh);
        auto out = output_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class MolecularDynamicsKernel>(range<1>(size), [=, problem_size = size, neighCount_ = neighCount,
                                                                         inum_ = inum, cutsq_ = cutsq, lj1_ = lj1, lj2_ = lj2](id<1> idx) {
            size_t gid = idx[0];

            if (gid < problem_size) {
                float4 ipos = in[gid];
                float4 f = {0.0f, 0.0f, 0.0f, 0.0f};
                int j = 0;
                while (j < neighCount_) {
                    int jidx = neigh[j * inum_ + gid];
                    float4 jpos = in[jidx];
                    float delx = ipos.x() - jpos.x();
                    float dely = ipos.y() - jpos.y();
                    float delz = ipos.z() - jpos.z();
                    float r2inv = delx * delx + dely * dely + delz * delz;
                    if (r2inv < cutsq_) {
                        r2inv = 10.0f / r2inv;
                        float r6inv = r2inv * r2inv * r2inv;
                        float forceC = r2inv * r6inv * (lj1_ * r6inv - lj2_);
                        f.x() += delx * forceC;
                        f.y() += dely * forceC;
                        f.z() += delz * forceC;
                    }
                    j++;
                }
                out[gid] = f;
            }
            });
        });
        q.wait();
        auto end = std::chrono::steady_clock::now();
        std::cout << "mod_dyn size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
        return 0;
}
