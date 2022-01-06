#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(conv3D);
using namespace cl::sycl;

void init(std::vector<float> A, int size) {
    const int NI = size;
    const int NJ = size;
    const int NK = size;
    for (int i = 0; i < NI; ++i) {
        for (int j = 0; j < NJ; ++j) {
            for (int k = 0; k < NK; ++k) {
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

    accelerator_selector as;
    queue q(as);
    // queue q;

    std::vector<float> A(size*size*size);
    init(A, size);
    std::vector<float> B(size*size*size);

    buffer<float> A_buff(A.data(), range<1>(size * size * size));
    buffer<float> B_buff(B.data(), range<1>(size * size * size));
    buffer<int> n_buff(&size, range<1>(1));
    
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read_write>(cgh);
        auto B_access = B_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class conv3D>([=]() {
            const int NI = no_access[0];
            const int NJ = no_access[0];
            const int NK = no_access[0];
            const float c11 = +2, c21 = +5, c31 = -8;
            const float c12 = -3, c22 = +6, c32 = -9;
            const float c13 = +4, c23 = +7, c33 = +10;
            for (int i = 1; i < NI - 1; ++i) {
                for (int j = 1; j < NJ - 1; ++j) {
                    for (int k = 1; k < NK - 1; ++k) {
                        B_access[i * (NK * NJ) + j * NK + k] = c11 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c21 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c31 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c12 * A_access[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A_access[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] + c32 * A_access[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A_access[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c13 * A_access[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A_access[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c23 * A_access[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A_access[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] + c33 * A_access[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
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