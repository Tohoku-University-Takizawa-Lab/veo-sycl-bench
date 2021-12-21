#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

REGISTER_KERNEL(correlation);
using namespace cl::sycl;

#define FLOAT_N 3214212.01
#define EPS 0.005
#define sqrt_of_array_cell(x, j) sqrt(x[j])

void init_arrays(std::vector<float> data_access, int size) {
    const int M = size;
    const int N = size;
    for (int i = 0; i <= M; i++) {
        for (int j = 0; j <= N; j++) {
            data_access[i * N + j] = ((float)i * j) / (M + 1);
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

    std::vector<float> data((size + 1) * (size + 1));
    init_arrays(data, size);
    std::vector<float> mean(size + 1);
    std::vector<float> stddev(size + 1);
    std::vector<float> symmat((size + 1) * (size + 1));

    buffer<float> data_buff(data.data(), range<1>((size + 1) * (size + 1)));
    buffer<float> mean_buff(mean.data(), range<1>(size + 1));
    buffer<float> stddev_buff(stddev.data(), range<1>(size + 1));
    buffer<float> symmat_buff(symmat.data(), range<1>((size + 1) * (size + 1)));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto data_access = data_buff.get_access<access::mode::read_write>(cgh);
        auto mean_access = mean_buff.get_access<access::mode::write>(cgh);
        auto stddev_access = stddev_buff.get_access<access::mode::write>(cgh);
        auto symmat_access = symmat_buff.get_access<access::mode::write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class correlation>([=]() {
            const int M = no_access[0];
            const int N = no_access[0];
            for (int j = 1; j <= M; j++) {
                mean_access[j] = 0.0;
                for (int i = 1; i <= N; i++) {
                    mean_access[j] += data_access[i * (M + 1) + j];
                }
                mean_access[j] /= (float)FLOAT_N;
            }
            for (int j = 1; j <= M; j++) {
                stddev_access[j] = 0.0;
                for (int i = 1; i <= N; i++) {
                    stddev_access[j] += (data_access[i * (M + 1) + j] - mean_access[j]) * (data_access[i * (M + 1) + j] - mean_access[j]);
                }
                stddev_access[j] /= FLOAT_N;
                stddev_access[j] = sqrt_of_array_cell(stddev_access, j);
                stddev_access[j] = stddev_access[j] <= EPS ? 1.0 : stddev_access[j];
            }
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= M; j++) {
                    data_access[i * (M + 1) + j] -= mean_access[j];
                    data_access[i * (M + 1) + j] /= sqrt(FLOAT_N);
                    data_access[i * (M + 1) + j] /= stddev_access[j];
                }
            }
            for (int j1 = 1; j1 <= M - 1; j1++) {
                symmat_access[j1 * (M + 1) + j1] = 1.0;
                for (int j2 = j1 + 1; j2 <= M; j2++) {
                    symmat_access[j1 * (M + 1) + j2] = 0.0;
                    for (int i = 1; i <= N; i++) {
                        symmat_access[j1 * (M + 1) + j2] += (data_access[i * (M + 1) + j1] * data_access[i * (M + 1) + j2]);
                    }
                    symmat_access[j2 * (M + 1) + j1] = symmat_access[j1 * (M + 1) + j2];
                }
            }
            symmat_access[M * (M + 1) + M] = 1.0;
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "correlation size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}