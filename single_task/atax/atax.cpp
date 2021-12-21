#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(atax);
using namespace cl::sycl;

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(std::vector<float> x, std::vector<float> A, int size) {
    const int NX = size;
    const int NY = size;
    for (int i = 0; i < NX; i++) {
        x[i] = i * M_PI;
        for (int j = 0; j < NY; j++) {
            A[i * NY + j] = ((float)i * (j)) / NX;
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

    int problem_bytes = size * size * sizeof(float);
    std::vector<float> x(size);
    std::vector<float> A(size * size);
    init_array(x, A, size);

    std::vector<float> y(size);
    std::vector<float> tmp(size);

    buffer<float> A_buff(A.data(), range<1>(size * size));
    buffer<float> x_buff(x.data(), range<1>(size));
    buffer<float> y_buff(y.data(), range<1>(size));
    buffer<float> tmp_buff(tmp.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto x_access = x_buff.get_access<access::mode::read>(cgh);
        auto y_access = y_buff.get_access<access::mode::write>(cgh);
        auto tmp_access = tmp_buff.get_access<access::mode::write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class atax>([=]() {
            const int NX = no_access[0];
            const int NY = no_access[0];
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NY; j++) {
                    tmp_access[i] += A_access[i * NY + j] * x[j];
                }
                for (int j = 0; j < NY; j++) {
                    y_access[j] += A_access[i * NY + j] * tmp_access[i];
                }
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "atax size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}