#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

using DATA_TYPE = float;

void init(DATA_TYPE* A, size_t size) {
    const auto NI = size;
    const auto NJ = size;
    const auto NK = size;
    for (size_t i = 0; i < NI; ++i) {
        for (size_t j = 0; j < NJ; ++j) {
            for (size_t k = 0; k < NK; ++k) {
                A[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
            }
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<DATA_TYPE> A(size * size * size);
    std::vector<DATA_TYPE> B(size * size * size);
    init(A.data(), size);

    buffer<DATA_TYPE, 3> A_buffer(A.data(), range<3>(size, size, size));
    buffer<DATA_TYPE, 3> B_buffer(B.data(), range<3>(size, size, size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto B = B_buffer.get_access<access::mode::discard_write>(cgh);

        cgh.parallel_for<class conv3D>(B_buffer.get_range(), [=, size_ = size](item<3> item) {
            const auto i = item[0];
            const auto j = item[1];
            const auto k = item[2];

            const DATA_TYPE c11 = +2, c21 = +5, c31 = -8;
            const DATA_TYPE c12 = -3, c22 = +6, c32 = -9;
            const DATA_TYPE c13 = +4, c23 = +7, c33 = +10;

            if ((i > 0) && (j > 0) && (k > 0) && (i < (size_ - 1)) && (j < (size_ - 1)) && (k < (size_ - 1))) {
                B[item] = c11 * A[{(i - 1), (j - 1), (k - 1)}] + c13 * A[{(i + 1), (j - 1), (k - 1)}] + c21 * A[{(i - 1), (j - 1), (k - 1)}] + c23 * A[{(i + 1), (j - 1), (k - 1)}] + c31 * A[{(i - 1), (j - 1), (k - 1)}] + c33 * A[{(i + 1), (j - 1), (k - 1)}] + c12 * A[{(i + 0), (j - 1), (k + 0)}] + c22 * A[{(i + 0), (j + 0), (k + 0)}] + c32 * A[{(i + 0), (j + 1), (k + 0)}] + c11 * A[{(i - 1), (j - 1), (k + 1)}] + c13 * A[{(i + 1), (j - 1), (k + 1)}] + c21 * A[{(i - 1), (j + 0), (k + 1)}] + c23 * A[{(i + 1), (j + 0), (k + 1)}] + c31 * A[{(i - 1), (j + 1), (k + 1)}] + c33 * A[{(i + 1), (j + 1), (k + 1)}];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "3DConvolution size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
