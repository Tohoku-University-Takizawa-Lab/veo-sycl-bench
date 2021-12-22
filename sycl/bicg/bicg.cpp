#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

void init_array(DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* r, size_t size) {
    const auto NX = size;
    const auto NY = size;
    for (size_t i = 0; i < NX; i++) {
        r[i] = i * M_PI;

        for (size_t j = 0; j < NY; j++) {
            A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
        }
    }
    for (size_t i = 0; i < NY; i++) {
        p[i] = i * M_PI;
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> A(size * size);
    std::vector<DATA_TYPE> r(size);
    std::vector<DATA_TYPE> s(size);
    std::vector<DATA_TYPE> p(size);
    std::vector<DATA_TYPE> q(size);
    init_array(A.data(), p.data(), r.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE> r_buffer(r.data(), range<1>(size));
    buffer<DATA_TYPE> s_buffer(s.data(), range<1>(size));
    buffer<DATA_TYPE> p_buffer(p.data(), range<1>(size));
    buffer<DATA_TYPE> q_buffer(q.data(), range<1>(size));

    queue device_queue;
    device_queue.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto r = r_buffer.get_access<access::mode::read>(cgh);
        auto s = s_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Bicg1>(s_buffer.get_range(), [=, size_ = size](item<1> item) {
            const auto j = item[0];
            for (size_t i = 0; i < size_; i++) {
                s[item] += A[{i, j}] * r[i];
            }
        });
    });
    device_queue.wait();
    device_queue.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto p = p_buffer.get_access<access::mode::read>(cgh);
        auto q = q_buffer.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<class Bicg2>(q_buffer.get_range(), [=, size_ = size](item<1> item) {
            const auto i = item[0];

            for (size_t j = 0; j < size_; j++) {
                q[item] += A[{i, j}] * p[j];
            }
        });
    });
    device_queue.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "bicg size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
