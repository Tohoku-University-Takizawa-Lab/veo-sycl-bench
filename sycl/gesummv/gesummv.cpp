#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = float;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, size_t size) {
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        x[i] = 1;
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = 2;
            B[i * N + j] = 3;
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
