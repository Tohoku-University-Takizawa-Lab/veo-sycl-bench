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

    std::vector<float> input1(size, 1.0);
    std::vector<float> input2(size, 2.0);
    std::vector<float> output(size, 0);

    buffer<float> input1_buf(input1.data(), range<1>(size));
    buffer<float> input2_buf(input2.data(), range<1>(size));
    buffer<float> output_buf(output.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto in1 = input1_buf.get_access<access::mode::read>(cgh);
        auto in2 = input2_buf.get_access<access::mode::read>(cgh);
        auto intermediate_product = output_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class ScalarProdKernel>(range<1>(size), [=](id<1> gid) {
            intermediate_product[gid] = in1[gid] * in2[gid];
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "saclar_prod size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
