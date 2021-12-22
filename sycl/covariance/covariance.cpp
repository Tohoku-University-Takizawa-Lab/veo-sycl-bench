#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = float;

constexpr DATA_TYPE float_n = 3214212.01;

void init_arrays(DATA_TYPE* data, size_t size) {
    const auto M = size;
    const auto N = size;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> data((size + 1) * (size + 1));
    std::vector<DATA_TYPE> mean(size + 1);
    std::vector<DATA_TYPE> symmat((size + 1) * (size + 1));
    init_arrays(data.data(), size);

    buffer<DATA_TYPE, 2> data_buffer(data.data(), range<2>(size + 1, size + 1));
    buffer<DATA_TYPE> mean_buffer(mean.data(), range<1>(size + 1));
    buffer<DATA_TYPE, 2> symmat_buffer(symmat.data(), range<2>(size + 1, size + 1));

    queue device_queue;
    device_queue.submit([&](handler& cgh) {
        auto data = data_buffer.get_access<access::mode::read>(cgh);
        auto mean = mean_buffer.get_access<access::mode::discard_write>(cgh);
        cgh.parallel_for<class CovarianceMean>(range<1>(size), id<1>(1), [=, N_ = size](item<1> item) {
            const auto j = item[0];
            for (size_t i = 1; i <= N_; i++) {
                mean[item] += data[{i, j}];
            }
            mean[item] /= float_n;
        });
    });
    device_queue.wait();
    device_queue.submit([&](handler& cgh) {
        auto mean = mean_buffer.get_access<access::mode::read>(cgh);
        auto data = data_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class CovarianceReduce>(range<2>(size, size), id<2>(1, 1), [=](item<2> item) {
            const auto j = item[1];
            data[item] -= mean[j];
        });
    });
    device_queue.wait();
    device_queue.submit([&](handler& cgh) {
        auto data = data_buffer.get_access<access::mode::read>(cgh);
        auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);
        auto symmat2 = symmat_buffer.get_access<access::mode::discard_write>(cgh);
        cgh.parallel_for<class CovarianceCovar>(range<1>(size), id<1>(1), [=, M_ = size, N_ = size](item<1> item) {
            const auto j1 = item[0];
            symmat[{j1, j1}] = 1.0;
            for (size_t j2 = j1; j2 <= M_; j2++) {
                symmat[{j1, j2}] = 0.0;
                for (size_t i = 1; i <= N_; i++) {
                    symmat[{j1, j2}] += data[{i, j1}] * data[{i, j2}];
                }
                symmat2[{j2, j1}] = symmat[{j1, j2}];
            }
        });
    });
    device_queue.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "covariance size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
