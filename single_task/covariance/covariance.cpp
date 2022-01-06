#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>


REGISTER_KERNEL(covariance);
using namespace cl::sycl;

void init_arrays(std::vector<float> data, int size) {
    const int M = size;
    const int N = size;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            data[i * (N + 1) + j] = ((float)i * j) / M;
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

    std::vector<float> data((size+1)*(size+1));
    init_arrays(data, size);
    std::vector<float> mean(size+1);
    std::vector<float> symmat((size+1)*(size+1));

    buffer<float> data_buff(data.data(), range<1>((size + 1) * (size + 1)));
    buffer<float> mean_buff(mean.data(), range<1>(size + 1));
    buffer<float> symmat_buff(symmat.data(), range<1>((size + 1) * (size + 1)));
    buffer<int> n_buff(&size, range<1>(1));
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto data_access = data_buff.get_access<access::mode::write>(cgh);
        auto symmat_access = symmat_buff.get_access<access::mode::write>(cgh);
        auto mean_access = mean_buff.get_access<access::mode::write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class covariance>([=]() {
            float float_n = 3214212.01;
            const auto M = no_access[0];
            const auto N = no_access[0];
            for (int j = 1; j <= M; j++) {
                mean_access[j] = 0.0;
                for (int i = 1; i <= N; i++) {
                    mean_access[j] += data_access[i * (M + 1) + j];
                }
                mean_access[j] /= float_n;
            }
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= M; j++) {
                    data_access[i * (M + 1) + j] -= mean_access[j];
                }
            }
            for (int j1 = 1; j1 <= M; j1++) {
                for (int j2 = j1; j2 <= M; j2++) {
                    symmat_access[j1 * (M + 1) + j2] = 0.0;
                    for (int i = 1; i <= N; i++) {
                        symmat_access[j1 * (M + 1) + j2] += data_access[i * (M + 1) + j1] * data_access[i * (M + 1) + j2];
                    }
                    symmat_access[j2 * (M + 1) + j1] = symmat_access[j1 * (M + 1) + j2];
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