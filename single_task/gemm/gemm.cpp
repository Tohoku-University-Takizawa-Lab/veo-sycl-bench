#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>


REGISTER_KERNEL(gemm);
using namespace cl::sycl;

#define ALPHA 32412
#define BETA 2123

void init(std::vector<float> A, std::vector<float> B, std::vector<float> C, int size) {
    const int NI = size;
    const int NJ = size;
    const int NK = size;
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NK; j++) {
            A[i * NK + j] = ((float)i * j) / NI;
        }
    }
    for (int i = 0; i < NK; i++) {
        for (int j = 0; j < NJ; j++) {
            B[i * NJ + j] = ((float)i * j + 1) / NJ;
        }
    }
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            C[i * NJ + j] = ((float)i * j + 2) / NJ;
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
    if(argc > 2 && argv[2] == std::string("vh")){
        q = queue();
        std::cout << "vh queue\n"; 
    }else{
        std::cout << "ve queue\n"; 
    }

    int problem_bytes = size * size * sizeof(float);
    std::vector<float> A(size*size);
    std::vector<float> B(size*size);
    std::vector<float> C(size*size);
    init(A, B, C, size);

    buffer<float> A_buff(A.data(), range<1>(size * size));
    buffer<float> B_buff(B.data(), range<1>(size * size));
    buffer<float> C_buff(C.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read_write>(cgh);
        auto B_access = B_buff.get_access<access::mode::read_write>(cgh);
        auto C_access = C_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class gemm>([=]() {
            const int NI = no_access[0];
            const int NJ = no_access[0];
            const int NK = no_access[0];
            for (int i = 0; i < NI; i++) {
                for (int j = 0; j < NJ; j++) {
                    C_access[i * NJ + j] *= BETA;
                    for (int k = 0; k < NK; ++k) {
                        C_access[i * NJ + j] += ALPHA * A_access[i * NK + k] * B_access[k * NJ + j];
                    }
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