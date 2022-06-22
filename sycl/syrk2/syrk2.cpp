#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

using DATA_TYPE = float;
constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void init_arrays(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
    const auto N = size;
    const auto M = size;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
        }
        for (size_t j = 0; j < M; j++) {
            A[i * N + j] = ((DATA_TYPE)i * j) / N;
            B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
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

    init_arrays(A.data(), B.data(), C.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> B_buffer(B.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> C_buffer(C.data(), range<2>(size, size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto B = B_buffer.get_access<access::mode::read>(cgh);
        auto C = C_buffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Syr2k1>(C_buffer.get_range(), [=, M_ = size](item<2> item) {
            const auto i = item[0];
            const auto j = item[1];
            C[item] *= BETA;
            for (size_t k = 0; k < M_; k++) {
                C[item] += ALPHA * A[{i, k}] * B[{j, k}] + ALPHA * B[{i, k}] * A[{j, k}];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "syrk2 size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
