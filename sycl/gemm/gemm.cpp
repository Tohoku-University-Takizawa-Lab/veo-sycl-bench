#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = float;

#define ALPHA 32412
#define BETA 2123

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
    std::vector<DATA_TYPE> C(size * size);

    init(A.data(), B.data(), C.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> B_buffer(B.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> C_buffer(C.data(), range<2>(size, size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto B = B_buffer.get_access<access::mode::read>(cgh);
        auto C = C_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Gemm>(C_buffer.get_range(), [=, NK_ = size](item<2> item) {
            const auto i = item[0];
            const auto j = item[1];
            C[item] *= BETA;
            for (size_t k = 0; k < NK_; k++) {
                C[item] += ALPHA * A[{i, k}] * B[{k, j}];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "gemm size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
