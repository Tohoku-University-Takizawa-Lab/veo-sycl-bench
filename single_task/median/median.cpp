#include "bitmap.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

REGISTER_KERNEL(median);
using namespace cl::sycl;

using namespace std;
void swap(Dot A[], int i, int j) {
    if (A[i].x > A[j].x) {
        float temp = A[i].x;
        A[i].x = A[j].x;
        A[j].x = temp;
    }
}
int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    vector<Dot> input(size*size);
    load_bitmap_mirrored("../../Brommy.bmp", size, input);
    std::vector<Dot> output(size*size);
    buffer<Dot> in_buff(input.data(), range<1>(size * size));
    buffer<Dot> out_buff(output.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto in_access = in_buff.get_access<access::mode::read>(cgh);
        auto out_access = out_buff.get_access<access::mode::write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class median>([=]() {
            for (int i = 0; i < no_access[0]; i++) {
                int x = i % no_access[0];
                int y = i / no_access[0];
                Dot window[9];
                int k = 0;
                for (int i = -1; i < 2; i++) {
                    for (int j = -1; j < 2; j++) {
                        unsigned int xs = fmin(fmax(x + j, 0), no_access[0] - 1); // borders are handled here with extended values
                        unsigned int ys = fmin(fmax(y + i, 0), no_access[0] - 1);
                        window[k] = in_access[xs + ys * no_access[0]];
                        k++;
                    }
                    swap(window, 0, 1);
                    swap(window, 2, 3);
                    swap(window, 0, 2);
                    swap(window, 1, 3);
                    swap(window, 1, 2);
                    swap(window, 4, 5);
                    swap(window, 7, 8);
                    swap(window, 6, 8);
                    swap(window, 6, 7);
                    swap(window, 4, 7);
                    swap(window, 4, 6);
                    swap(window, 5, 8);
                    swap(window, 5, 7);
                    swap(window, 5, 6);
                    swap(window, 0, 5);
                    swap(window, 0, 4);
                    swap(window, 1, 6);
                    swap(window, 1, 5);
                    swap(window, 1, 4);
                    swap(window, 2, 7);
                    swap(window, 3, 8);
                    swap(window, 3, 7);
                    swap(window, 2, 5);
                    swap(window, 2, 4);
                    swap(window, 3, 6);
                    swap(window, 3, 5);
                    swap(window, 3, 4);
                }
                out_access[i] = window[4];
            }
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "median size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
