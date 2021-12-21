#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
REGISTER_KERNEL(gesummv);
using namespace cl::sycl;

void init(std::vector<float> A, std::vector<float> B, std::vector<float> x, int size) {
    const int N = size;
    for (int i = 0; i < N; i++) {
        x[i] = 1;
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 2;
            B[i * N + j] = 3;
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    std::vector<float> A(size * size);
    std::vector<float> B(size * size);
    std::vector<float> x(size);
    init(A, B, x, size);
    std::vector<float> y(size);
    std::vector<float> tmp(size);

    buffer<float> A_buff(A.data(), range<1>(size * size));
    buffer<float> B_buff(B.data(), range<1>(size * size));
    buffer<float> x_buff(x.data(), range<1>(size));
    buffer<float> y_buff(y.data(), range<1>(size));
    buffer<float> tmp_buff(tmp.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto B_access = B_buff.get_access<access::mode::read>(cgh);
        auto x_access = x_buff.get_access<access::mode::read>(cgh);
        auto y_access = y_buff.get_access<access::mode::read_write>(cgh);
        auto tmp_access = tmp_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class gesummv>([=]() {
            const int N = no_access[0];
            float alpha = 1;
            float beta = 1;
            for (int i = 0; i < N; i++) {
                tmp_access[i] = 0;
                y_access[i] = 0;
                for (int j = 0; j < N; j++) {
                    tmp_access[i] = A_access[i * N + j] * x_access[j] + tmp_access[i];
                    y_access[i] = B_access[i * N + j] * x_access[j] + y_access[i];
                }
                y_access[i] = alpha * tmp_access[i] + beta * y_access[i];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "vec_add size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}