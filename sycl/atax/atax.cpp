#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

void init_array(DATA_TYPE* x, DATA_TYPE* A, size_t size) {
    const auto NX = size;
    const auto NY = size;
    for (size_t i = 0; i < NX; i++) {
        x[i] = i * M_PI;
        for (size_t j = 0; j < NY; j++) {
            A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> A(size * size);
    std::vector<DATA_TYPE> x(size);
    std::vector<DATA_TYPE> y(size);
    std::vector<DATA_TYPE> tmp(size);
    init_array(x.data(), A.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE> x_buffer(x.data(), range<1>(size));
    buffer<DATA_TYPE> y_buffer(y.data(), range<1>(size));
    buffer<DATA_TYPE> tmp_buffer(tmp.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto x = x_buffer.get_access<access::mode::read>(cgh);
        auto tmp = tmp_buffer.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<class Atax1>(tmp_buffer.get_range(), [=, size_ = size](item<1> item) {
            const auto i = item[0];
            for (size_t j = 0; j < size_; j++) {
                tmp[item] += A[{i, j}] * x[j];
            }
        });
    });
    q.wait();
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto y = y_buffer.get_access<access::mode::read_write>(cgh);
        auto tmp = tmp_buffer.get_access<access::mode::read>(cgh);
        cgh.parallel_for<class Atax2>(y_buffer.get_range(), [=, size_ = size](item<1> item) {
            const auto j = item[0];
            for (size_t i = 0; i < size_; i++) {
                y[item] += A[{i, j}] * tmp[i];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "atax size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
