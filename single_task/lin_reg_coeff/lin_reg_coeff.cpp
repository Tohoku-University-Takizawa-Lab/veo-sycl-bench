#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(coeff);
using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    std::vector<float> input1(size, 1.0);
    std::vector<float> input2(size, 2.0);
    float sumv1 = 0;
    float sumv2 = 0;
    float xy = 0;
    float xx = 0;
    buffer<float> in1_buff(input1.data(), range<1>(size));
    buffer<float> in2_buff(input2.data(), range<1>(size));
    buffer<float> s1_buff(&sumv1, range<1>(1));
    buffer<float> s2_buff(&sumv2, range<1>(1));
    buffer<float> xy_buff(&xy, range<1>(1));
    buffer<float> xx_buff(&xx, range<1>(1));
    buffer<int> n_buff(&size, range<1>(1));
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto in1_access = in1_buff.get_access<access::mode::read>(cgh);
        auto in2_access = in2_buff.get_access<access::mode::read>(cgh);
        auto s1_access = s1_buff.get_access<access::mode::read_write>(cgh);
        auto s2_access = s2_buff.get_access<access::mode::read_write>(cgh);
        auto xy_access = xy_buff.get_access<access::mode::read_write>(cgh);
        auto xx_access = xx_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class coeff>([=]() {
            for (int i = 0; i < no_access[0]; ++i) {
                s1_access[0] += input1[i];
                s2_access[0] += input2[i];
                xy_access[0] += input1[i] * input2[i];
                xx_access[0] += input1[i] * input1[i];
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