#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

using DATA_TYPE = float;
constexpr DATA_TYPE alpha = 123;
constexpr DATA_TYPE beta = 14512;

void init_arrays(DATA_TYPE* A, DATA_TYPE* C, size_t size) {
    const auto N = size;
    const auto M = size;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            A[i * M + j] = ((DATA_TYPE)i * j) / N;
        }
        for (size_t j = 0; j < N; j++) {
            C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> A(size * size);
    std::vector<DATA_TYPE> C(size * size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> C_buffer(C.data(), range<2>(size, size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto C = C_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Syr2k2>(C_buffer.get_range(), [=, M_ = size](item<2> item) {
            const auto i = item[0];
            const auto j = item[1];
            C[item] *= beta;

            for (size_t k = 0; k < M_; k++) {
                C[item] += alpha * A[{i, k}] * A[{j, k}];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "syrk size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
