#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<float> input1(size);
    std::vector<float> input2(size);
    std::vector<float> output(size, 0);
    for (size_t i = 0; i < size; i++) {
        input1[i] = static_cast<float>(i);
        input2[i] = static_cast<float>(i);
    }

    buffer<float> input1_buf(input1.data(), range<1>(size));
    buffer<float> input2_buf(input2.data(), range<1>(size));
    buffer<float> output_buf(output.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto in1 = input1_buf.get_access<access::mode::read>(cgh);
        auto in2 = input2_buf.get_access<access::mode::read>(cgh);
        auto out = output_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class VecAddKernel>(range<1>(size), [=](id<1> gid) {
            out[gid] = in1[gid] + in2[gid];
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "vec_add size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
