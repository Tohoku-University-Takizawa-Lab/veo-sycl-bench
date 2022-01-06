#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(mvt);
using namespace cl::sycl;

void init_arrays(std::vector<float> a, std::vector<float> x1, std::vector<float> x2, std::vector<float> y_1, std::vector<float> y_2, int size) {
    const int N = size;
    for (int i = 0; i < N; i++) {
        x1[i] = 0.0;
        x2[i] = 0.0;
        y_1[i] = 0.0;
        y_2[i] = 0.0;
        for (int j = 0; j < N; j++) {
            a[i * N + j] = (float)(i + j + 1.0) / N;
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

    std::vector<float> a(size*size);
    std::vector<float> x1(size);
    std::vector<float> x2(size);
    std::vector<float> y1(size);
    std::vector<float> y2(size);
    init_arrays(a, x1, x2, y1, y2, size);

    buffer<float> a_buff(a.data(), range<1>(size * size));
    buffer<float> x1_buff(x1.data(), range<1>(size));
    buffer<float> x2_buff(x2.data(), range<1>(size));
    buffer<float> y1_buff(y1.data(), range<1>(size));
    buffer<float> y2_buff(y2.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto a_access = a_buff.get_access<access::mode::read_write>(cgh);
        auto x1_access = x1_buff.get_access<access::mode::read_write>(cgh);
        auto x2_access = x2_buff.get_access<access::mode::read_write>(cgh);
        auto y1_access = y1_buff.get_access<access::mode::read_write>(cgh);
        auto y2_access = y2_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class mvt>([=]() {
            const int N = no_access[0];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    x1_access[i] = x1_access[i] + a_access[i * N + j] * y1_access[j];
                }
            }
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    x2_access[k] = x2_access[k] + a_access[k * N + l] * y2_access[l];
                }
            }
        });
    });
    q.wait();
    auto task_end = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    std::cout << "size=" << size 
    << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" 
    << " task(" << std::chrono::duration_cast<std::chrono::milliseconds>(task_end - task_start).count() << ")" 
    << std::endl;
    return 0;
}