#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(mm2);
using namespace cl::sycl;

void init_array(std::vector<float> A, std::vector<float> B, std::vector<float> C, std::vector<float> D, int size) {
    int NI = size;
    int NJ = size;
    int NK = size;
    int NL = size;
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NK; j++) {
            A[i * NI + j] = ((float)i * j) / NI;
        }
    }
    for (int i = 0; i < NK; i++) {
        for (int j = 0; j < NJ; j++) {
            B[i * NK + j] = ((float)i * (j + 1)) / NJ;
        }
    }
    for (int i = 0; i < NL; i++) {
        for (int j = 0; j < NJ; j++) {
            C[i * NL + j] = ((float)i * (j + 3)) / NL;
        }
    }
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NL; j++) {
            D[i * NL + j] = ((float)i * (j + 2)) / NK;
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

    std::vector<float> A(size * size);
    std::vector<float> B(size * size);
    std::vector<float> C(size * size);
    std::vector<float> D(size * size);
    init_array(A, B, C, D, size);
    std::vector<float> E(size * size);
    buffer<float> A_buff(A.data(), range<1>(size * size));
    buffer<float> B_buff(B.data(), range<1>(size * size));
    buffer<float> C_buff(C.data(), range<1>(size * size));
    buffer<float> D_buff(D.data(), range<1>(size * size));
    buffer<float> E_buff(E.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));

    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read_write>(cgh);
        auto B_access = B_buff.get_access<access::mode::read_write>(cgh);
        auto C_access = C_buff.get_access<access::mode::read_write>(cgh);
        auto D_access = D_buff.get_access<access::mode::read_write>(cgh);
        auto E_access = E_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class mm2>([=]() {
            int NI = no_access[0];
            int NJ = no_access[0];
            int NK = no_access[0];
            int NL = no_access[0];
            for (int i = 0; i < NI; i++) {
                for (int j = 0; j < NJ; j++) {
                    for (int k = 0; k < NK; ++k) {
                        C_access[i * NJ + j] += A_access[i * NK + k] * B_access[k * NJ + j];
                    }
                }
            }
            for (int i = 0; i < NI; i++) {
                for (int j = 0; j < NL; j++) {
                    E_access[i * NL + j] = 0;
                    for (int k = 0; k < NJ; ++k) {
                        E_access[i * NL + j] += C_access[i * NJ + k] * D_access[k * NL + j];
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