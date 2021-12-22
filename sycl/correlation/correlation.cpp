#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

#define FLOAT_N 3214212.01
#define EPS 0.005

#define sqrt_of_array_cell(x, j) sqrt(x[j])

using DATA_TYPE = float;

void init_arrays(DATA_TYPE* data, size_t size) {
    const auto M = size;
    const auto N = size;

    for (size_t i = 0; i <= M; i++) {
        for (size_t j = 0; j <= N; j++) {
            data[i * N + j] = ((DATA_TYPE)i * j) / (M + 1);
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
    std::vector<DATA_TYPE> stddev(size + 1);
    std::vector<DATA_TYPE> symmat((size + 1) * (size + 1));
    init_arrays(data.data(), size);

    buffer<DATA_TYPE, 2> data_buffer(data.data(), range<2>(size + 1, size + 1));
    buffer<DATA_TYPE> mean_buffer(mean.data(), range<1>(size + 1));
    buffer<DATA_TYPE> stddev_buffer(stddev.data(), range<1>(size + 1));
    buffer<DATA_TYPE, 2> symmat_buffer(symmat.data(), range<2>(size + 1, size + 1));

    queue device_queue;
    device_queue.submit([&](handler& cgh) {
        auto data = data_buffer.get_access<access::mode::read>(cgh);
        auto mean = mean_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class CorrelationMean>(range<1>(size), id<1>(1), [=, N_ = size](item<1> item) {
            const auto j = item[0];
            for (size_t i = 1; i <= N_; i++) {
                mean[item] += data[{i, j}];
            }
            mean[item] /= ((DATA_TYPE)FLOAT_N);
        });
    });
    device_queue.wait();
    device_queue.submit([&](handler& cgh) {
        auto data = data_buffer.get_access<access::mode::read>(cgh);
        auto mean = mean_buffer.get_access<access::mode::read>(cgh);
        auto stddev = stddev_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class CorrelationStd>(range<1>(size), id<1>(1), [=, N_ = size](item<1> item) {
            const auto j = item[0];
            for (size_t i = 1; i <= N_; i++) {
                stddev[item] += (data[{i, j}] - mean[item]) * (data[{i, j}] - mean[item]);
            }
            stddev[item] /= FLOAT_N;
            stddev[item] = cl::sycl::sqrt(stddev[item]);
            stddev[item] = stddev[item] <= EPS ? 1.0 : stddev[item];
        });
    });
    device_queue.wait();
    device_queue.submit([&](handler& cgh) {
        auto data = data_buffer.get_access<access::mode::read_write>(cgh);
        auto mean = mean_buffer.get_access<access::mode::read>(cgh);
        auto stddev = stddev_buffer.get_access<access::mode::read>(cgh);
        cgh.parallel_for<class CorrelationReduce>(range<2>(size, size), id<2>(1, 1), [=](item<2> item) {
            const auto j = item[1];
            data[item] -= mean[j];
            data[item] /= cl::sycl::sqrt(FLOAT_N);
            data[item] /= stddev[j];
        });
    });
    device_queue.wait();
    device_queue.submit([&](handler& cgh) {
        auto data = data_buffer.get_access<access::mode::read>(cgh);
        auto symmat = symmat_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class CorrelationCorr>(range<1>(size), id<1>(1), [=, M_ = size, N_ = size](item<1> item) {
            // if(item[0] >= M_ - 1) return;
            const auto j1 = item[0];
            symmat[{j1, j1}] = 1.0;
            for (size_t j2 = j1 + 1; j2 <= M_; j2++) {
                symmat[{j1, j2}] = 0.0;
                for (size_t i = 1; i <= N_; i++) {
                    symmat[{j1, j2}] += data[{i, j1}] * data[{i, j2}];
                }
                symmat[{j2, j1}] = symmat[{j1, j2}];
            }
        });
    });
    device_queue.wait();
    device_queue.submit([&](handler& cgh) {
        auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);
        cgh.parallel_for<class Correlation5>(range<2>(1, 1), id<2>(size, size), [=](item<2> item) {
            symmat[item] = 1.0;
        });
    });
    device_queue.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "correlation size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
