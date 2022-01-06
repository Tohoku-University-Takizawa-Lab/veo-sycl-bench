#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(syrk2);
using namespace cl::sycl;

void init_arrays(std::vector<float> A, std::vector<float> B, std::vector<float> C, int size) {
    const int N = size;
    const int M = size;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = ((float)i * j + 2) / N;
        }
        for (int j = 0; j < M; j++) {
            A[i * N + j] = ((float)i * j) / N;
            B[i * N + j] = ((float)i * j + 1) / N;
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
    std::vector<float> C(size * size);
    init_arrays(A, B, C, size);
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

        cgh.single_task<class syrk2>([=]() {
            const int N = no_access[0];
            const int M = no_access[0];
            float alpha = 1;
            float beta = 1;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C_access[i * N + j] *= beta;
                }
            }
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < M; k++) {
                        C_access[i * N + j] += alpha * A_access[i * M + k] * B_access[j * M + k];
                        C_access[i * N + j] += alpha * B_access[i * M + k] * A_access[j * M + k];
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