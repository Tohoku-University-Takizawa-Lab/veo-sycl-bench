#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(fdtd2d);
using namespace cl::sycl;

int TMAX = 500;
void init_arrays(std::vector<float> fict, std::vector<float> ex, std::vector<float> ey, std::vector<float> hz, int size) {
    const int NX = size;
    const int NY = size;
    for (int i = 0; i < TMAX; i++) {
        fict[i] = (float)i;
    }
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            ex[i * NY + j] = ((float)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((float)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((float)(i - 9) * (j + 4) + 3) / NX;
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

    std::vector<float> fict(TMAX);
    std::vector<float> ex(size * (size + 1));
    std::vector<float> ey(size * (size + 1));
    std::vector<float> hz(size * size);
    init_arrays(fict, ex, ey, hz, size);

    buffer<float> fict_buff(fict.data(), range<1>(TMAX));
    buffer<float> ex_buff(ex.data(), range<1>((size + 1) * size));
    buffer<float> ey_buff(ey.data(), range<1>((size + 1) * size));
    buffer<float> hz_buff(hz.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto fict_access = fict_buff.get_access<access::mode::read_write>(cgh);
        auto ex_access = ex_buff.get_access<access::mode::read_write>(cgh);
        auto ey_access = ey_buff.get_access<access::mode::read_write>(cgh);
        auto hz_access = hz_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class fdtd2d>([=]() {
            const int NX = no_access[0];
            const int NY = no_access[0];
            const int TMAX = 500;
            for (int t = 0; t < TMAX; t++) {
                for (int j = 0; j < NY; j++) {
                    ey_access[0 * NY + j] = fict_access[t];
                }
                for (int i = 1; i < NX; i++) {
                    for (int j = 0; j < NY; j++) {
                        ey_access[i * NY + j] = ey_access[i * NY + j] - 0.5 * (hz_access[i * NY + j] - hz_access[(i - 1) * NY + j]);
                    }
                }
                for (int i = 0; i < NX; i++) {
                    for (int j = 1; j < NY; j++) {
                        ex_access[i * (NY + 1) + j] = ex_access[i * (NY + 1) + j] - 0.5 * (hz_access[i * NY + j] - hz_access[i * NY + (j - 1)]);
                    }
                }
                for (int i = 0; i < NX; i++) {
                    for (int j = 0; j < NY; j++) {
                        hz_access[i * NY + j] = hz_access[i * NY + j] - 0.7 * (ex_access[i * (NY + 1) + (j + 1)] - ex_access[i * (NY + 1) + j] + ey_access[(i + 1) * NY + j] - ey_access[i * NY + j]);
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