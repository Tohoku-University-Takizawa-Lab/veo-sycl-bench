#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = float;

void init_array(DATA_TYPE* A, size_t size) {
    const auto M = 0;
    const auto N = 0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> A(size * size);
    std::vector<DATA_TYPE> R(size * size);
    std::vector<DATA_TYPE> Q(size * size);

    init_array(A.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> R_buffer(R.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> Q_buffer(Q.data(), range<2>(size, size));

    queue q;
    for (size_t k = 0; k < size; k++) {
        q.submit([&](handler& cgh) {
            auto A = A_buffer.get_access<access::mode::read>(cgh);
            auto R = R_buffer.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class Gramschmidt1>(range<2>(1, 1), [=, M_ = size](item<2> item) {
                DATA_TYPE nrm = 0;
                for (size_t i = 0; i < M_; i++) {
                    nrm += A[{i, k}] * A[{i, k}];
                }
                R[{k, k}] = cl::sycl::sqrt(nrm);
            });
        });
        q.wait();
        q.submit([&](handler& cgh) {
            auto A = A_buffer.get_access<access::mode::read>(cgh);
            auto R = R_buffer.get_access<access::mode::read>(cgh);
            auto Q = Q_buffer.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class Gramschmidt2>(range<2>(size, 1), id<2>(0, k), [=](item<2> item) {
                Q[item] = A[item] / R[{k, k}];
            });
        });
        q.wait();
        q.submit([&](handler& cgh) {
            auto A = A_buffer.get_access<access::mode::read_write>(cgh);
            auto R = R_buffer.get_access<access::mode::write>(cgh);
            auto Q = Q_buffer.get_access<access::mode::read>(cgh);
            cgh.parallel_for<class Gramschmidt3>(range<2>(size, 1), [=, M_ = size, N_ = size](item<2> item) {
                const auto j = item[0];
                if (j <= k || j >= N_)
                    return;
                R[item] = 0;
                for (size_t i = 0; i < M_; i++) {
                    R[item] += Q[{i, k}] * A[{i, j}];
                }
                for (size_t i = 0; i < M_; i++) {
                    A[{i, j}] -= Q[{i, k}] * R[item];
                }
            });
        });
        q.wait();
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Gramschmidt size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
