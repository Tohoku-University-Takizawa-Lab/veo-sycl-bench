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
    std::vector<float> alpha(size);
    std::vector<float> beta(size);
    std::vector<float> output(size, 0);
    for (size_t i = 0; i < size; i++) {
        input1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        input2[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        alpha[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        beta[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    buffer<float> input1_buf(input1.data(), range<1>(size));
    buffer<float> input2_buf(input2.data(), range<1>(size));
    buffer<float> alpha_buf(alpha.data(), range<1>(size));
    buffer<float> beta_buf(beta.data(), range<1>(size));
    buffer<float> output_buf(output.data(), range<1>(size));

    queue q;
    q.submit([&](handler& cgh) {
        auto in1 = input1_buf.template get_access<access::mode::read>(cgh);
        auto in2 = input2_buf.template get_access<access::mode::read>(cgh);
        auto alpha = alpha_buf.template get_access<access::mode::read>(cgh);
        auto beta = beta_buf.template get_access<access::mode::read>(cgh);
        auto output = output_buf.template get_access<access::mode::discard_write>(cgh);
        range<1> ndrange(size);
        cgh.parallel_for<class LinearRegressionKernel>(ndrange,
                                                       [=, problem_size = size](id<1> idx) {
                                                           size_t gid = idx[0];
                                                           float a = alpha[gid];
                                                           float b = beta[gid];
                                                           float error = 0.0;
                                                           if (gid < problem_size) {
                                                               // Use parallel reduction to add errors
                                                               for (size_t i = 0; i < problem_size; i++) {
                                                                   float e = (a * in1[i] + b) - in2[i];
                                                                   error += e * e;
                                                               }
                                                           }
                                                           output[gid] = error;
                                                       });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "lin_reg_error size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
