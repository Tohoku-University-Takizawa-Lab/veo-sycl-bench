#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(conv2D);
using namespace cl::sycl;

void init(std::vector<float> A, int size) {
    const int NI = size;
    const int NJ = size;
    for (int i = 0; i < NI; ++i) {
        for (int j = 0; j < NJ; ++j) {
            A[i * NJ + j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

int main(const int argc, const char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    if(argc > 2 && argv[2] == std::string("vh")){
        q = queue();
        std::cout << "vh queue\n"; 
    }else{
        std::cout << "ve queue\n"; 
    }

    std::vector<float> A(size*size);
    init(A, size);

    buffer<float> A_buff(A.data(), range<1>(size * size));
    std::vector<float> B(size*size);
    buffer<float> B_buff(B.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));
    
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read_write>(cgh);
        auto B_access = B_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class conv2D>([=]() {
            const int NI = no_access[0];
            const int NJ = no_access[0];
            const float c11 = +0.2, c21 = +0.5, c31 = -0.8;
            const float c12 = -0.3, c22 = +0.6, c32 = -0.9;
            const float c13 = +0.4, c23 = +0.7, c33 = +0.10;
            for (int i = 1; i < NI - 1; ++i) {
                for (int j = 1; j < NJ - 1; ++j) {
                    B_access[i * NJ + j] = c11 * A_access[(i - 1) * NJ + (j - 1)] + c12 * A_access[(i + 0) * NJ + (j - 1)] + c13 * A_access[(i + 1) * NJ + (j - 1)] + c21 * A_access[(i - 1) * NJ + (j + 0)] + c22 * A_access[(i + 0) * NJ + (j + 0)] + c23 * A_access[(i + 1) * NJ + (j + 0)] + c31 * A_access[(i - 1) * NJ + (j + 1)] + c32 * A_access[(i + 0) * NJ + (j + 1)] + c33 * A_access[(i + 1) * NJ + (j + 1)];
                }
            }
        });
    });
    q.wait();
    auto task_end = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    std::cout << "2DCon size=" << size 
    << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" 
    << " task(" << std::chrono::duration_cast<std::chrono::milliseconds>(task_end - task_start).count() << ")" 
    << std::endl;
    return 0;
}