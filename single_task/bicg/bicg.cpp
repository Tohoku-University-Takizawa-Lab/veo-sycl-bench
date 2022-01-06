#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(bicg);
using namespace cl::sycl;

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(std::vector<float> A, std::vector<float> p, std::vector<float> r, int size) {
    const int NX = size;
    const int NY = size;
    for (int i = 0; i < NX; i++) {
        r[i] = i * M_PI;
        for (int j = 0; j < NY; j++) {
            A[i * NY + j] = ((float)i * j) / NX;
        }
    }
    for (int i = 0; i < NY; i++) {
        p[i] = i * M_PI;
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
    std::vector<float> p(size);
    std::vector<float> r(size);
    init_array(A, p, r, size);

    std::vector<float> s(size);
    std::vector<float> Q(size);

    buffer<float> A_buff(A.data(), range<1>(size * size));
    buffer<float> p_buff(p.data(), range<1>(size));
    buffer<float> r_buff(r.data(), range<1>(size));
    buffer<float> s_buff(s.data(), range<1>(size));
    buffer<float> Q_buff(Q.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));

    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read_write>(cgh);
        auto r_access = r_buff.get_access<access::mode::read_write>(cgh);
        auto s_access = s_buff.get_access<access::mode::read_write>(cgh);
        auto p_access = p_buff.get_access<access::mode::read_write>(cgh);
        auto Q_access = Q_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class bicg>([=]() {
            const int NX = no_access[0];
            const int NY = no_access[0];
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NY; j++) {
                    s_access[j] += r_access[i] * A_access[i * NY + j];
                    Q_access[i] += A_access[i * NY + j] * p_access[j];
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