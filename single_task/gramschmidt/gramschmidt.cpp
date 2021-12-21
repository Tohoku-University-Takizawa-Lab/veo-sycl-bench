#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

REGISTER_KERNEL(gramschmidt);
using namespace cl::sycl;

void init_array(std::vector<float> A, int size) {
    const int M = 0;
    const int N = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = ((float)(i + 1) * (j + 1)) / (M + 1);
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
    // queue q;

    std::vector<float> A(size * size);
    std::vector<float> R(size * size);
    std::vector<float> Q(size * size);
    init_array(A, size);
    buffer<float> A_buff(A.data(), range<1>(size * size));
    buffer<float> R_buff(R.data(), range<1>(size * size));
    buffer<float> Q_buff(Q.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto A_access = A_buff.get_access<access::mode::read_write>(cgh);
        auto R_access = R_buff.get_access<access::mode::read_write>(cgh);
        auto Q_access = Q_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class gramschmidt>([=]() {
            const int M = no_access[0];
            const int N = no_access[0];
            for (int k = 0; k < N; k++) {
                float nrm = 0;
                for (int i = 0; i < M; i++) {
                    nrm += A_access[i * N + k] * A_access[i * N + k];
                }
                R_access[k * N + k] = sqrt(nrm);
                for (int i = 0; i < M; i++) {
                    Q_access[i * N + k] = A_access[i * N + k] / R_access[k * N + k];
                }
                for (int j = k + 1; j < N; j++) {
                    R_access[k * N + j] = 0;
                    for (int i = 0; i < M; i++) {
                        R_access[k * N + j] += Q_access[i * N + k] * A_access[i * N + j];
                    }
                    for (int i = 0; i < M; i++) {
                        A_access[i * N + j] = A_access[i * N + j] - Q_access[i * N + k] * R_access[k * N + j];
                    }
                }
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "vec_add size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}