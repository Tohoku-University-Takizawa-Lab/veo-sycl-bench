#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

using DATA_TYPE = float;

void init(DATA_TYPE* A, size_t size) {
    const auto NI = size;
    const auto NJ = size;

    for (size_t i = 0; i < NI; ++i) {
        for (size_t j = 0; j < NJ; ++j) {
            A[i * NJ + j] = (float)rand() / (float)RAND_MAX;
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
    init(A.data(), size);

    buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(size, size));
    buffer<DATA_TYPE, 2> B_buffer(B.data(), range<2>(size, size));

    queue q;
    q.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto B = B_buffer.get_access<access::mode::discard_write>(cgh);

        cgh.parallel_for<class conv2D>(B_buffer.get_range(), [=, size_ = size](item<2> item) {
            const auto i = item[0];
            const auto j = item[1];

            const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
            const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
            const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;
            if ((i > 0) && (j > 0) && (i < size_ - 1) && (j < size_ - 1)) {
                B[item] = c11 * A[{(i - 1), (j - 1)}] + c12 * A[{(i + 0), (j - 1)}] + c13 * A[{(i + 1), (j - 1)}] + c21 * A[{(i - 1), (j + 0)}] + c22 * A[{(i + 0), (j + 0)}] + c23 * A[{(i + 1), (j + 0)}] + c31 * A[{(i - 1), (j + 1)}] + c32 * A[{(i + 0), (j + 1)}] + c33 * A[{(i + 1), (j + 1)}];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "2DConvolution size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
