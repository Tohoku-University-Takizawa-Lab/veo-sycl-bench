#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = float;

void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2, size_t size) {
    const auto N = size;
    for (size_t i = 0; i < N; i++) {
        x1[i] = 0.0;
        x2[i] = 0.0;
        y_1[i] = 0.0;
        y_2[i] = 0.0;
        for (size_t j = 0; j < N; j++) {
            a[i * N + j] = (DATA_TYPE)(i + j + 1.0) / N;
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> a(size * size);
    std::vector<DATA_TYPE> x1(size);
    std::vector<DATA_TYPE> x2(size);
    std::vector<DATA_TYPE> y1(size);
    std::vector<DATA_TYPE> y2(size);
    init_arrays(a.data(), x1.data(), x2.data(), y1.data(), y2.data(), size);

    buffer<DATA_TYPE, 2> a_buffer(a.data(), range<2>(size, size));
    buffer<DATA_TYPE> x1_buffer(x1.data(), range<1>(size));
    buffer<DATA_TYPE> x2_buffer(x2.data(), range<1>(size));
    buffer<DATA_TYPE> y1_buffer(y1.data(), range<1>(size));
    buffer<DATA_TYPE> y2_buffer(y2.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto a = a_buffer.get_access<access::mode::read>(cgh);
        auto y1 = y1_buffer.get_access<access::mode::read>(cgh);
        auto x1 = x1_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Mvt1>(x1_buffer.get_range(), [=, N_ = size](item<1> item) {
            const auto i = item[0];
            for (size_t j = 0; j < N_; j++) {
                x1[i] += a[{i, j}] * y1[j];
            }
        });
    });
    q.wait();
    q.submit([&](handler& cgh) {
        auto a = a_buffer.get_access<access::mode::read>(cgh);
        auto y2 = y2_buffer.get_access<access::mode::read>(cgh);
        auto x2 = x2_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Mvt2>(x1_buffer.get_range(), [=, N_ = size](item<1> item) {
            const auto k = item[0];
            for (size_t l = 0; l < N_; l++) {
                x2[k] += a[{k, l}] * y2[l];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "mvt size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
