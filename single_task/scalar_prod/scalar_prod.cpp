#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(prod);
using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    long problem_bytes = size * sizeof(float);
    std::vector<float> input1(size, 1.0);
    std::vector<float> input2(size, 2.0);
    std::vector<float> output(size, 0);
    buffer<float> in1_buff(input1.data(), range<1>(size));
    buffer<float> in2_buff(input2.data(), range<1>(size));
    buffer<float> out_buff(output.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto in1_access = in1_buff.get_access<access::mode::read_write>(cgh);
        auto in2_access = in2_buff.get_access<access::mode::read_write>(cgh);
        auto out_access = out_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class prod>([=]() {
            for (int i = 0; i < no_access[0]; ++i) {
                out_access[i] = in1_access[i] * in2_access[i];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "scalar_prod size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}