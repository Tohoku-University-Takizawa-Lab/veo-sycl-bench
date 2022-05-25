#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = double;

constexpr auto TMAX = 500;

void init_arrays(DATA_TYPE* fict, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t size) {
    const auto NX = size;
    const auto NY = size;
    for (size_t i = 0; i < TMAX; i++) {
        fict[i] = (DATA_TYPE)i;
    }
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
    }
}
int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> fict(TMAX);
    std::vector<DATA_TYPE> ex(size * (size + 1));
    std::vector<DATA_TYPE> ey((size+1) *size);
    std::vector<DATA_TYPE> hz(size * size);

    init_arrays(fict.data(), ex.data(), ey.data(), hz.data(), size);

    buffer<DATA_TYPE> fict_buffer(fict.data(), range<1>(TMAX));
    buffer<DATA_TYPE, 2> ex_buffer(ex.data(), range<2>(size, size + 1));
    buffer<DATA_TYPE, 2> ey_buffer(ey.data(), range<2>(size + 1, size));
    //buffer<DATA_TYPE, 2> hz_buffer(hz.data(), range<2>(size + 1, size));
    buffer<DATA_TYPE, 2> hz_buffer(hz.data(), range<2>(size, size));

    queue device_queue;
    for (size_t t = 0; t < TMAX; t++) {
        device_queue.submit([&](handler& cgh) {
            auto fict = fict_buffer.get_access<access::mode::read>(cgh);
            auto ey = ey_buffer.get_access<access::mode::read_write>(cgh);
            auto hz = hz_buffer.get_access<access::mode::read>(cgh);
            cgh.parallel_for<class Fdtd2d1>(range<2>(size, size), [=](item<2> item) {
                const auto i = item[0];
                const auto j = item[1];
                if (i == 0) {
                    ey[item] = fict[t];
                } else {
                    ey[item] = ey[item] - 0.5 * (hz[item] - hz[{(i - 1), j}]);
                }
            });
        });
        device_queue.wait();
        device_queue.submit([&](handler& cgh) {
            auto ex = ex_buffer.get_access<access::mode::read_write>(cgh);
            auto hz = hz_buffer.get_access<access::mode::read>(cgh);
           cgh.parallel_for<class Fdtd2d2>(range<2>(size, size), [=, NX_ = size, NY_ = size](item<2> item) {
                const auto i = item[0];
                const auto j = item[1];
                if (j > 0)
                    ex[item] = ex[item] - 0.5 * (hz[item] - hz[{i, (j - 1)}]);
            });
        });
        device_queue.wait();
        device_queue.submit([&](handler& cgh) {
            auto ex = ex_buffer.get_access<access::mode::read>(cgh);
            auto ey = ey_buffer.get_access<access::mode::read>(cgh);
            auto hz = hz_buffer.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<class Fdtd2d3>(hz_buffer.get_range(), [=](item<2> item) {
                const auto i = item[0];
                const auto j = item[1];
                hz[item] = hz[item] - 0.7 * (ex[{i, (j + 1)}] - ex[item] + ey[{(i + 1), j}] - ey[item]);
            });
        });
        device_queue.wait();
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "fdtd2d size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
