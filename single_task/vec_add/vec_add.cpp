#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(add);
using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    if(argc > 2 && argv[2] == std::string("vh")){
        q = queue();
    }
    // queue q;
    std::vector<int> input1(size);
    std::vector<int> input2(size);
    std::vector<int> output(size, 0);
    for (int i = 0; i < size; i++) {
        input1[i] = i;
        input2[i] = i;
    }

    buffer<int> in1_buff(input1.data(), range<1>(size));
    buffer<int> in2_buff(input2.data(), range<1>(size));
    buffer<int> out_buff(output.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));

    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto in1_access = in1_buff.get_access<access::mode::read_write>(cgh);
        auto in2_access = in2_buff.get_access<access::mode::read_write>(cgh);
        auto out_access = out_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class add>([=]() {
            for (int i = 0; i < no_access[0]; ++i) {
                out_access[i] = in1_access[i] + in2_access[i];
            }
        });
    });
    q.wait();
    auto task_end = std::chrono::steady_clock::now();
    // for (int i = 0; i < size; ++i) {
    //     std::cout << input1[i] << " + " << input2[i] << " = " << output[i] << std::endl;
    // }
    auto end = std::chrono::steady_clock::now();
    std::cout << "vec_add size=" << size 
    << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" 
    << " task(" << std::chrono::duration_cast<std::chrono::milliseconds>(task_end - task_start).count() << ")" 
    << std::endl;
    return 0;
}