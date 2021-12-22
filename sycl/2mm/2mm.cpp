#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using DATA_TYPE = float;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, size_t size) {
    const auto NI = size;
    const auto NJ = size;
    const auto NK = size;
    const auto NL = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NK; j++) {
            A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
        }
    }
    for (size_t i = 0; i < NK; i++) {
        for (size_t j = 0; j < NJ; j++) {
            B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
        }
    }
    for (size_t i = 0; i < NL; i++) {
        for (size_t j = 0; j < NJ; j++) {
            C[i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<float> A(size * size);
    std::vector<float> B(size * size);
    std::vector<float> C(size * size);
    std::vector<float> D(size * size);
    std::vector<float> E(size * size);
    init_array(A.data(), B.data(), C.data(), D.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> B_buffer(B.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> C_buffer(C.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> D_buffer(D.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> E_buffer(E.data(), range<2>(size, size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto B = B_buffer.get_access<access::mode::read>(cgh);
        auto C = C_buffer.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<class Polybench_2mm_1>(C_buffer.get_range(), [=, size_ = size](item<2> item) {
            const auto i = item[0];
            const auto j = item[1];
            for (size_t k = 0; k < size_; k++) {
                C[item] += A[{i, k}] * B[{k, j}];
            }
        });
    });
    q.wait();
    q.submit([&](handler& cgh) {
        auto C = C_buffer.get_access<access::mode::read>(cgh);
        auto D = D_buffer.get_access<access::mode::read>(cgh);
        auto E = E_buffer.get_access<access::mode::discard_write>(cgh);

        cgh.parallel_for<class Polybench_2mm_2>(E_buffer.get_range(), [=, size_ = size](item<2> item) {
            const auto i = item[0];
            const auto j = item[1];
            E[item] = 0;
            for (size_t k = 0; k < size_; k++) {
                E[item] += C[{i, k}] * D[{k, j}];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "2mm size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
