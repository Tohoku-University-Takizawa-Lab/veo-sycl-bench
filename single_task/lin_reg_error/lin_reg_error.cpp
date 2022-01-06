#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

REGISTER_KERNEL(error);
using namespace cl::sycl;
using namespace std;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    vector<float> input1(size);
    vector<float> input2(size);
    vector<float> alpha(size);
    vector<float> beta(size);
    for (int i = 0; i < size; i++) {
        input1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        input2[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        alpha[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        beta[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    vector<float> output(size, 0);
    buffer<float> input1_buf(input1.data(), range<1>(size));
    buffer<float> input2_buf(input2.data(), range<1>(size));
    buffer<float> alpha_buf(alpha.data(), range<1>(size));
    buffer<float> beta_buf(beta.data(), range<1>(size));
    buffer<float> output_buf(output.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto in1 = input1_buf.get_access<access::mode::read>(cgh);
        auto in2 = input2_buf.get_access<access::mode::read>(cgh);
        auto al = alpha_buf.get_access<access::mode::read>(cgh);
        auto be = beta_buf.get_access<access::mode::read>(cgh);
        auto out = output_buf.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class error>([=]() {
            for (int i = 0; i < no_access[0]; i++) {
                float error = 0.0;
                for (int j = 0; j < no_access[0]; j++) {
                    float e = (al[i] * in1[j] + be[i]) - in2[j];
                    error += e * e;
                }
                out[i] = error;
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
