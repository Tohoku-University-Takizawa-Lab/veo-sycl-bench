#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(syrk);
using namespace cl::sycl;

void init_arrays(std::vector<float> A, std::vector<float> C, int size) {
    const int N = size;
    const int M = size;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i * M + j] = ((float)i * j) / N;
        }
        for (int j = 0; j < N; j++) {
            C[i * M + j] = ((float)i * j + 2) / N;
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

    std::vector<float> A(size*size);
    std::vector<float> C(size*size);
    init_arrays(A, C, size);
    buffer<float> A_buff(A.data(), range<1>(size * size));
    buffer<float> C_buff(C.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read>(cgh);
        auto C_access = C_buff.get_access<access::mode::write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class syrk>([=]() {
            const int N = no_access[0];
            const int M = no_access[0];
            float alpha = 123;
            float beta = 14512;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C_access[i * M + j] *= beta;
                }
            }
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < M; k++) {
                        C_access[i * N + j] += alpha * A_access[i * M + k] * A_access[j * M + k];
                    }
                }
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "syrk size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}