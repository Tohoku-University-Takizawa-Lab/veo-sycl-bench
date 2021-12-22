#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = float;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
    const auto NI = size;
    const auto NJ = size;
    const auto NK = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NK; j++) {
            A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
        }
    }
    for (size_t i = 0; i < NK; i++) {
        for (size_t j = 0; j < NJ; j++) {
            B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> A(size * size);
    std::vector<DATA_TYPE> B(size * size);
    std::vector<DATA_TYPE> x(size);
    std::vector<DATA_TYPE> y(size);
    std::vector<DATA_TYPE> tmp(size);

    init(A.data(), B.data(), x.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> B_buffer(B.data(), range<2>(size, size));
    buffer<DATA_TYPE> x_buffer(x.data(), range<1>(size));
    buffer<DATA_TYPE> y_buffer(y.data(), range<1>(size));
    buffer<DATA_TYPE> tmp_buffer(tmp.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto B = B_buffer.get_access<access::mode::read>(cgh);
        auto x = x_buffer.get_access<access::mode::read>(cgh);
        auto y = y_buffer.get_access<access::mode::read_write>(cgh);
        auto tmp = tmp_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Gesummv>(y.get_range(), [=, N_ = size](item<1> item) {
            const auto i = item[0];
            for (size_t j = 0; j < N_; j++) {
                tmp[item] += A[{i, j}] * x[j];
                y[item] += B[{i, j}] * x[j];
            }
            y[item] = ALPHA * tmp[item] + BETA * y[item];
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "gesummv size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
